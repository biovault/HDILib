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

std::map<SPIRVShader, std::vector<uint32_t>>& getSPIRVBinaries();