#include "shader_progs.h"

std::vector<float> BoundsShaderProg::compute(float padding) {
    // Implement the compute function using _mgr and _tensors
    // This is a placeholder implementation
    std::vector<uint32_t> result;
    // Actual computation logic goes here
    const std::vector<std::shared_ptr<kp::Memory>> params = {
        _tensors[Tensors::BOUNDS],
        _tensors[Tensors::POSITION],
        _tensors[Tensors::NUM_POINTS]
    };
    auto algorithm = _mgr.algorithm(params, _shaderBinary, kp::Workgroup({1,1,1}), {}, std::vector<float>({padding}));
    auto seq = _mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>(params)
        ->eval();


    return *_tensors[Tensors::BOUNDS]->data<std::vector<float>>();
}

std::vector<uint8_t> StencilShaderProg::compute(uint32_t width, uint32_t height, unsigned int num_points, std::vector<float> bounds) {
    auto npwidth = width;
    if (npwidth % 8) {
        npwidth += 8 - (npwidth % 8); // Align to next multiple of 8
    }
    
    auto stencil_array = std::vector<uint8_t>(npwidth * height, 4);
    auto stencil_out = _mgr.imageT(stencil_array, npwidth, height, 4 );
    auto pushConsts = std::vector<float>({ bounds[0], bounds[1], bounds[2], bounds[3], (float)width, (float)height});

    const std::vector<std::shared_ptr<kp::Memory>> params = {_tensors[Tensors::POSITION], stencil_out }
    auto algorithm = _mgr.algorithm(params, _shaderBinary, kp::Workgroup({ num_points, 1, 1 }), {}, pushConsts);
    auto seq = _mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>(params)
        ->eval();

    return stencil_out.data<uint8_t>();
}

std::vector<float> FieldComputationShaderProg::compute(std::vector<uint8_t> stencil, uint32_t width, uint32_t height) {
    auto npwidth = width;
    if (npwidth % 8) {
        npwidth += 8 - (npwidth % 8); // Align to next multiple of 8
    }

    auto stencil_in = _mgr.imageT(stencil, npwidth, height, 4);
    auto field_out = _mgr.imageT(std::vector<float>(npwidth * height * 4, 0.0f), npwidth, height, 4);
    const std::vector<std::shared_ptr<kp::Memory>> params = {
        _tensors[Tensors::POSITION],
        _tensors[Tensors::BOUNDS],
        field_out,
        stencil_in,
        _tensors[Tensors::NUM_POINTS]
    };
    auto pushConsts = std::vector<float>({ (float)width, (float)height, _function_support });
    auto algorithm = _mgr.algorithm(params, _shaderBinary, kp::Workgroup({ width, height, 1 }), {}, pushConsts);
    auto seq = _mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>(params)
        ->eval();

    return field_out->data<float();
}

void InterpolationShaderProg::compute(std::vector<float> fields, uint32_t width, uint32_t height) {
    auto npwidth = width;
    if (npwidth % 8) {
        npwidth += 8 - (npwidth % 8); // Align to next multiple of 8
    }

    auto fields_in = _mgr.imageT(fields, npwidth, height, 4);
    const std::vector<std::shared_ptr<kp::Memory>> params = {
        _tensors[Tensors::POSITION],
        _tensors[Tensors::INTERP_FIELDS],
        _tensors[Tensors::SUM_Q],
        _tensors[Tensors::BOUNDS],
        _tensors[Tensors::NUM_POINTS]
        fields_in
    };

    auto pushConsts = std::vector<float>({ (float)width, (float)height});
    auto algorithm = _mgr.algorithm(params, _shaderBinary, kp::Workgroup({ 1, 1, 1 }), {}, pushConsts);
    auto seq = _mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>(params)
        ->eval();

    _sum_Q = _tensors[Tensors::SUM_Q]->data<float>()[0];
}

float ForcesShaderProg::compute(unsigned int num_points, float exaggeration) {
    const std::vector<std::shared_ptr<kp::Memory>> params = {
        _tensors[Tensors::POSITION],
        _tensors[Tensors::NEIGHBOUR],
        _tensors[Tensors::INDEX],
        _tensors[Tensors::PROBABILITIES],
        _tensors[Tensors::INTERP_FIELDS],
        _tensors[Tensors::GRADIENTS],
        _tensors[Tensors::NUM_POINTS],
        _tensors[Tensors::KLDIV]
    };

    auto sum_Q = _tensors[Tensors::SUM_Q]->data<float>()[0];
    auto pushConsts = std::vector<float>({ exaggeration, sum_Q });
    auto grid_size unsigned int(sqrt(num_points) + 1);
    auto algorithm = _mgr.algorithm(params, _shaderBinary, kp::Workgroup({ grid_size, grid_size, 1 }), {}, pushConsts);
    auto seq = _mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>(params)
        ->eval();

    return _tensors[Tensors::KLDIV]->data<float>()[0];
}

void UpdateShaderProg::void compute(unsigned int num_points, float eta, float minimum_gain, float iteration, float momentum, unsigned int momentumn_switch, float final_momentum, float gain_mult) {
    const std::vector<std::shared_ptr<kp::Memory>> params = {
        _tensors[Tensors::POSITION],
        _tensors[Tensors::GRADIENTS],
        _tensors[Tensors::PREV_GRADIENTS],
        _tensors[Tensors::GAIN],
        _tensors[Tensors::NUM_POINTS]
    };

    auto num_workgroups = unsigned int((num_points * 2 / 64) + 1);
    auto grid_size = unsigned int(sqrt(num_workgroups) + 1);
    auto pushConsts = std::vector<float>({ eta, minimum_gain, float(iteration), float(momentum_switch), momentum, final_momentum, gain_mult });
    auto algorithm = _mgr.algorithm(params, _shaderBinary, kp::Workgroup({ grid_size, grid_size, 1 }), {}, pushConsts);
    auto seq = _mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>(params)
        ->eval();
}

void CenterScaleProg::compute(unsigned int num_points, float exaggeration) {
    const std::vector<std::shared_ptr<kp::Memory>> params = {
        _tensors[Tensors::POSITION],
        _tensors[Tensors::BOUNDS],
        _tensors[Tensors::NUM_POINTS]
    };
    auto scale = 0.0f;
    auto diameter = 0.0f;

    if exaggeration > 1.2:
        scale = 1.0f;
        diameter = 0.1;

    auto pushConsts = std::vector<float>({ scale, diameter });
    auto num_workgroups = unsigned int(num_points / 128) + 1
    auto grid_size = unsigned int(sqrt(num_workgroups) + 1);
    auto algorithm = _mgr.algorithm(params, _shaderBinary, kp::Workgroup({ grid_size, grid_size, 1 }), {}, pushConsts);
    auto seq = _mgr.sequence()
        ->record<kp::OpSyncDevice>(params)
        ->record<kp::OpAlgoDispatch>(algorithm)
        ->record<kp::OpSyncLocal>(params)
        ->eval();
}