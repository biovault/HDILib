#pragma once
#include <map>
#include <vector>
#include <memory>
#include <kompute/Kompute.hpp>

// PushConstant structs
struct boundsPushConstants {
  float padding;
  uint32_t num_points;
};

// The UBO structs used to create the buffers
struct stencilParams {
  float rect_min[2];
  float rect_max[2];  
  float image_size[2];
};

struct fieldParams {
  uint32_t num_points;
  float image_size[2];
  float support;
};

struct fieldParamsPF1 {
  uint32_t num_points;
  uint32_t num_workgroups;
  float image_size[2];
  float support;
};

struct fieldParamsPF2 {
  uint32_t num_workgroups;
};

struct interpParams {
  uint32_t num_points;
  float image_size[2];
};

struct interp2Params {
  uint32_t num_workgroups;
};

struct forcesParams {
  uint32_t num_points;
  float exaggeration;
};

struct updaterParams {
  uint32_t num_points;
  float eta;
  float minGain;
  float iter;
  float mom_iter;
  float mom;
  float final_mom;
  float mult;
};

struct centerScaleParams {
  uint32_t num_points;
  float scale;
  float diameter;
};

enum class ShaderBuffers {
  POSITION = 0,
  NEIGHBOUR = 1,
  PROBABILITIES = 2,
  INDEX = 3,
  INTERP_FIELDS = 4,
  GRADIENTS = 5,
  PREV_GRADIENTS = 6,
  GAIN = 7,
  BOUNDS = 8,
  SUM_Q = 9,
  KLDIV = 10,
  NUM_POINTS = 11,
  IMAGE_WORKGROUP = 12,
  UBO_STENCIL = 13,
  UBO_FIELD = 14,
  UBO_INTERP = 15,
  UBO_INTERP2 = 16,
  UBO_FORCES = 17,
  UBO_UPDATE = 18,
  UBO_CENTER_SCALE = 19,
  DEBUG = 20, // For any debug output buffers
  PARTIAL_SUM = 21,
  ATOMIC_COUNTER = 22,
  UBO_FIELD_PF_1 = 23,
  UBO_FIELD_PF_2 = 24
};

typedef std::map<ShaderBuffers, std::shared_ptr<kp::Tensor>> TensorMap;