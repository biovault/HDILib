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
    {SPIRVShader::BOUNDS, std::vector<uint32_t>(bounds::BOUNDS_COMP_SPV.begin(), bounds::BOUNDS_COMP_SPV.end())},
    {SPIRVShader::COMPUTE_FIELDS, std::vector<uint32_t>(compute_fields::COMPUTE_FIELDS_COMP_SPV.begin(), compute_fields::COMPUTE_FIELDS_COMP_SPV.end())},
    {SPIRVShader::COMPUTE_FORCES, std::vector<uint32_t>(compute_forces::COMPUTE_FORCES_COMP_SPV.begin(), compute_forces::COMPUTE_FORCES_COMP_SPV.end())},
    {SPIRVShader::INTERP_FIELDS, std::vector<uint32_t>(interp_fields::INTERP_FIELDS_COMP_SPV.begin(), interp_fields::INTERP_FIELDS_COMP_SPV.end())},
    {SPIRVShader::STENCIL, std::vector<uint32_t>(stencil::STENCIL_COMP_SPV.begin(), stencil::STENCIL_COMP_SPV.end())},
    {SPIRVShader::UPDATE, std::vector<uint32_t>(update::UPDATE_COMP_SPV.begin(), update::UPDATE_COMP_SPV.end())},
    {SPIRVShader::CENTER_SCALE, std::vector<uint32_t>(center_scale::CENTER_SCALE_COMP_SPV.begin(), center_scale::CENTER_SCALE_COMP_SPV.end())}
};