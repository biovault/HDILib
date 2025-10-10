#pragma once
#include <map>
#include <vector>
#include <memory>
#include <kompute/Kompute.hpp>

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
};

typedef std::map<ShaderBuffers, std::shared_ptr<kp::Tensor>> TensorMap;