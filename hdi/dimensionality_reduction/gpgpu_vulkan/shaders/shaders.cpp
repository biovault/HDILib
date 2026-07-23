#include "shaders.h"
std::vector<uint32_t> getSPIRVBinaries(SPIRVShader shader) {
  switch(shader) {
    case SPIRVShader::BOUNDS: 
      return {bounds::BOUNDS_COMP_SPV.begin(), bounds::BOUNDS_COMP_SPV.end()};
    case SPIRVShader::COMPUTE_FIELDS:
      return {compute_fields::COMPUTE_FIELDS_COMP_SPV.begin(), compute_fields::COMPUTE_FIELDS_COMP_SPV.end()};
    case SPIRVShader::COMPUTE_FIELDS_ENH:
      return {compute_fields_enh::COMPUTE_FIELDS_ENH_COMP_SPV.begin(), compute_fields_enh::COMPUTE_FIELDS_ENH_COMP_SPV.end()};
    case SPIRVShader::COMPUTE_FORCES:
      return {compute_forces::COMPUTE_FORCES_COMP_SPV.begin(), compute_forces::COMPUTE_FORCES_COMP_SPV.end()};
    case SPIRVShader::INTERP_FIELDS:
      return {interp_fields::INTERP_FIELDS_COMP_SPV.begin(), interp_fields::INTERP_FIELDS_COMP_SPV.end()};
    case SPIRVShader::INTERP_FIELDS_ENH1:
      return {interp1::INTERP1_COMP_SPV.begin(), interp1::INTERP1_COMP_SPV.end()};
    case SPIRVShader::INTERP_FIELDS_ENH2:
      return {interp2::INTERP2_COMP_SPV.begin(), interp2::INTERP2_COMP_SPV.end()};
    case SPIRVShader::STENCIL:
      return {stencil::STENCIL_COMP_SPV.begin(), stencil::STENCIL_COMP_SPV.end()};
    case SPIRVShader::RESET_COUNTER:
      return {reset_counter::RESET_COUNTER_COMP_SPV.begin(), reset_counter::RESET_COUNTER_COMP_SPV.end()};
    case SPIRVShader::STENCIL2ACTIVE:
      return {stencil2active::STENCIL2ACTIVE_COMP_SPV.begin(), stencil2active::STENCIL2ACTIVE_COMP_SPV.end()};
    case SPIRVShader::FIELD_WORKGROUP:
      return {compute_field_workgroup::COMPUTE_FIELD_WORKGROUP_COMP_SPV.begin(), compute_field_workgroup::COMPUTE_FIELD_WORKGROUP_COMP_SPV.end()};
    case SPIRVShader::UPDATE:
      return {update::UPDATE_COMP_SPV.begin(), update::UPDATE_COMP_SPV.end()};
    case SPIRVShader::CENTER_SCALE:
      return {center_scale::CENTER_SCALE_COMP_SPV.begin(), center_scale::CENTER_SCALE_COMP_SPV.end()};
    default:
      return {};
  }
}