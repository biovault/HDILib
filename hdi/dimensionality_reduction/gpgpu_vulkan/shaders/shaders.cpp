#include "shaders.h"
std::map<SPIRVShader, std::vector<uint32_t>>& getSPIRVBinaries() {
  static auto SPIRVShaderBinaries = std::map<SPIRVShader, std::vector<uint32_t>>({
    {SPIRVShader::BOUNDS, std::vector<uint32_t>(bounds::BOUNDS_COMP_SPV.begin(), bounds::BOUNDS_COMP_SPV.end())},
    {SPIRVShader::COMPUTE_FIELDS, std::vector<uint32_t>(compute_fields::COMPUTE_FIELDS_COMP_SPV.begin(), compute_fields::COMPUTE_FIELDS_COMP_SPV.end())},
    {SPIRVShader::COMPUTE_FIELDS_ENH, std::vector<uint32_t>(compute_fields_enh::COMPUTE_FIELDS_ENH_COMP_SPV.begin(), compute_fields_enh::COMPUTE_FIELDS_ENH_COMP_SPV.end())},
    {SPIRVShader::COMPUTE_FORCES, std::vector<uint32_t>(compute_forces::COMPUTE_FORCES_COMP_SPV.begin(), compute_forces::COMPUTE_FORCES_COMP_SPV.end())},
    {SPIRVShader::INTERP_FIELDS, std::vector<uint32_t>(interp_fields::INTERP_FIELDS_COMP_SPV.begin(), interp_fields::INTERP_FIELDS_COMP_SPV.end())},
    {SPIRVShader::INTERP_FIELDS_ENH1, std::vector<uint32_t>(interp1::INTERP1_COMP_SPV.begin(), interp1::INTERP1_COMP_SPV.end())},
    {SPIRVShader::INTERP_FIELDS_ENH2, std::vector<uint32_t>(interp2::INTERP2_COMP_SPV.begin(), interp2::INTERP2_COMP_SPV.end())},
    {SPIRVShader::STENCIL, std::vector<uint32_t>(stencil::STENCIL_COMP_SPV.begin(), stencil::STENCIL_COMP_SPV.end())},
    {SPIRVShader::STENCIL2ACTIVE, std::vector<uint32_t>(stencil2active::STENCIL2ACTIVE_COMP_SPV.begin(), stencil2active::STENCIL2ACTIVE_COMP_SPV.end())},
    {SPIRVShader::UPDATE, std::vector<uint32_t>(update::UPDATE_COMP_SPV.begin(), update::UPDATE_COMP_SPV.end())},
    {SPIRVShader::CENTER_SCALE, std::vector<uint32_t>(center_scale::CENTER_SCALE_COMP_SPV.begin(), center_scale::CENTER_SCALE_COMP_SPV.end())}
    });
  return SPIRVShaderBinaries;
}