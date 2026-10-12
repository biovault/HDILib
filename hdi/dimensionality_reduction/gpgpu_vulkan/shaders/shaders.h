#pragma once

#include <map>
#include <vector>

#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/bounds.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/compute_fields.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/compute_fields_enh.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/compute_forces.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/interp_fields.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/interp1.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/interp2.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/stencil.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/reset_counter.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/stencil2active.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/compute_field_workgroup.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/update.hpp"
#include "hdi/dimensionality_reduction/gpgpu_vulkan/shaders/center_scale.hpp"

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

std::map<SPIRVShader, std::vector<uint32_t>>& getSPIRVBinaries();