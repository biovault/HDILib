#pragma once
#include <map>
#include <vector>
#include "bounds.hpp"
#include "compute_fields.hpp"
#include "compute_fields_enh.hpp"
#include "compute_forces.hpp"
#include "interp_fields.hpp" 
#include "interp1.hpp"
#include "interp2.hpp"
#include "stencil.hpp"
#include "reset_counter.hpp"
#include "stencil2active.hpp"
#include "compute_field_workgroup.hpp"
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
  INTERP_FIELDS_ENH1 = 7,
  INTERP_FIELDS_ENH2 = 8,
  STENCIL2ACTIVE = 9,
  COMPUTE_FIELDS_ENH = 10,
  RESET_COUNTER = 11,
  FIELD_WORKGROUP = 12,
};

std::vector<uint32_t> getSPIRVBinaries(SPIRVShader shader);