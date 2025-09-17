#pragma once
#include <map>
#include <vector>
#include "bounds.hpp"
#include "compute_fields.hpp"
#include "compute_forces.hpp"
#include "interp_fields.hpp"    
#include "stencil.hpp"
#include "update.hpp"   
#include "center_scale.hpp"

enum class SPIRVShader {
    BOUNDS = 0,
    COMPUTE_FIELDS = 1,
    COMPUTE_FORCES = 2,
    INTERP_FIELDS = 3,
    STENCIL = 4,
    UPDATE = 5,
    CENTER_SCALE = 6,
};

std::map<SPIRVShader, std::vector<uint32_t>> SPIRVShaderBinaries = {
    {BOUNDS, std::vector<uint32_t>(bounds::SHADER_COMP_SPV.begin(), bounds::SHADER_COMP_SPV.end())},
    {COMPUTE_FIELDS, std::vector<uint32_t>(compute_fields::SHADER_COMP_SPV.begin(), compute_fields::SHADER_COMP_SPV.end())},
    {COMPUTE_FORCES, std::vector<uint32_t>(compute_forces::SHADER_COMP_SPV.begin(), compute_forces::SHADER_COMP_SPV.end())},
    {INTERP_FIELDS, std::vector<uint32_t>(interp_fields::SHADER_COMP_SPV.begin(), interp_fields::SHADER_COMP_SPV.end())},
    {STENCIL, std::vector<uint32_t>(stencil::SHADER_COMP_SPV.begin(), stencil::SHADER_COMP_SPV.end())},
    {UPDATE, std::vector<uint32_t>(update::SHADER_COMP_SPV.begin(), update::SHADER_COMP_SPV.end())},
    {CENTER_SCALE, std::vector<uint32_t>(center_scale::SHADER_COMP_SPV.begin(), center_scale::SHADER_COMP_SPV.end())}
};