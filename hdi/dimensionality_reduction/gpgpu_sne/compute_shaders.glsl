#define GLSL(version, shader)  "#version " #version "\n" #shader

// Sample the static field texture for all the points.
// and sum the value to product the normalization factor Z-hat
// see https://arxiv.org/pdf/1805.10817 Eq 13
// This is subsequently used to calculate the repulsive forces Eq 14.
const char* interp_fields_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer Val { vec4 Values[]; };
  layout(std430, binding = 2) buffer SumB { float Sum[]; };
  layout(std430, binding = 3) buffer BoundsInterface { vec2 Bounds[]; };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  shared float reduction_array[64];

  uniform sampler2D fields;
  uniform uint num_points;

  void main() {
    uint lid = gl_LocalInvocationIndex.x;
    uint groupSize = gl_WorkGroupSize.x;

    vec2 min_bounds = Bounds[0];
    vec2 max_bounds = Bounds[1];
    vec2 range = max_bounds - min_bounds;
    vec2 inv_range = 1.0 / range;

    float sum_Q = 0;
    for (uint i = lid; i < num_points; i += groupSize)
    {
      // Position of point in range 0 to 1
      vec2 point = (Positions[i] - min_bounds) * inv_range;

      // Bilinearly sample the input texture 
      // using GPU native texture operation
      // see https://arxiv.org/pdf/1805.10817 5.1.2 
      vec4 v = texture(fields, point);
      // SumQ (Z-hat) is the sum of the (static field -1)
      // that is used as the normalization factor for 
      // the repulsive forces.
      sum_Q += max(v.x - 1, 0.0);
      Values[i] = v;
    }

 
    // Reduce add sum_Q to a single value
    //uint reduction_size = 64;
    if (lid >= 64) {
      reduction_array[lid - 64] = sum_Q;
    }
    barrier();
    if (lid < 64) {
      reduction_array[lid] += sum_Q;
    }
    barrier();
    if (lid < 32) {
      reduction_array[lid] += reduction_array[lid + 32];
    }
    barrier();
    if (lid < 16) {
      reduction_array[lid] += reduction_array[lid + 16];
    }
    barrier();
    if (lid < 8) {
      reduction_array[lid] += reduction_array[lid + 8];
    }
    barrier();
    if (lid < 4) {
      reduction_array[lid] += reduction_array[lid + 4];
    }
    barrier();
    if (lid < 2) {
      reduction_array[lid] += reduction_array[lid + 2];
    }
    barrier();
    if (lid < 1) {
      Sum[0] = reduction_array[0] + reduction_array[1];
    }
  }
);

// Calculate the forces acting on a point i
// this is the modified (by the exaggeration factor on the attractive force)
// sum of the attractive forces (Eq 12) and the repulsive forces (Eq 14) 
const char* compute_forces_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer Neigh { uint Neighbours[]; };
  layout(std430, binding = 2) buffer Prob { float Probabilities[]; };
  layout(std430, binding = 3) buffer Ind { int Indices[]; };
  layout(std430, binding = 4) buffer Fiel { vec4 Fields[]; };
  layout(std430, binding = 5) buffer Grad { vec2 Gradients[]; };
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

  const uint group_size = 64;
  shared vec2 sum_positive_red[group_size];

  //layout(rg32f) uniform image2D point_tex;
  uniform uint num_points;
  uniform float exaggeration;
  uniform float sum_Q; // Z-hat from the previous interpolation step

  void main() {
    uint i = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint groupSize = gl_WorkGroupSize.x;
    uint lid = gl_LocalInvocationID.x;

    float inv_num_points = 1.0 / float(num_points);
    float inv_sum_Q = 1.0 / sum_Q;

    if (i >= num_points)
      return;

    // Get the point coordinates
    vec2 point_i = Positions[i];

    //computing positive forces
    vec2 sum_positive = vec2(0);

    int index = Indices[i * 2 + 0];
    int size = Indices[i * 2 + 1];

    vec2 positive_force = vec2(0);
    for (uint j = lid; j < size; j += group_size) {
      // Get other point coordinates
      vec2 point_j = Positions[Neighbours[index + j]];

      // Calculate 2D distance between the two points
      vec2 dist = point_i - point_j;

      // Similarity measure of the two points - https://arxiv.org/pdf/1805.10817 Eq 5
      float qij = 1 / (1 + dist.x*dist.x + dist.y*dist.y);

      // Calculate the attractive force  - https://arxiv.org/pdf/1805.10817 Eq 12
      positive_force += Probabilities[index + j] * qij * dist * inv_num_points;
    }

    // Reduce add sum_positive_red to a single value
    if (lid >= 32) {
      sum_positive_red[lid - 32] = positive_force;
    }
    barrier();
    if (lid < 32) {
      sum_positive_red[lid] += positive_force;
    }
    for (uint reduceSize = group_size/4; reduceSize > 1; reduceSize /= 2)
    {
      barrier();
      if (lid < reduceSize) {
        sum_positive_red[lid] += sum_positive_red[lid + reduceSize];
      }
    }
    barrier();
    if (lid < 1) {
      sum_positive = sum_positive_red[0] + sum_positive_red[1];

      // Computing repulsive forces - https://arxiv.org/pdf/1805.10817 Eq 14
      vec2 sum_negative = Fields[i].yz * inv_sum_Q;

      Gradients[i] = 4 * (exaggeration * sum_positive - sum_negative);
    }
  }
);

// Update the positions of the embedding points based on the gradients
// This is the point update step - https://arxiv.org/pdf/1805.10817 5.1.3
const char* update_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ float Positions[]; };
  layout(std430, binding = 1) buffer GradientLayout { float Gradients[]; };
  layout(std430, binding = 2) buffer PrevGradientLayout { float PrevGradients[]; };
  layout(std430, binding = 3) buffer GainLayout { float Gain[]; };
  layout(local_size_x = 64, local_size_y = 1, local_size_z = 1) in;

  uniform uint num_points;
  uniform float eta;
  uniform float minGain;
  uniform float iter;
  uniform float mom_iter;
  uniform float mom;
  uniform float final_mom;
  uniform float mult;

  void main() {
    uint workGroupID = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint i = workGroupID * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i >= num_points * 2)
      return;

    float grad = Gradients[i];
    float pgrad = PrevGradients[i];
    float gain = Gain[i];

    gain = sign(grad) != sign(pgrad) ? gain + 0.2 : gain * 0.8;
    gain = max(gain, minGain);

    float etaGain = eta * gain;
    grad = (grad > 0 ? 1 : -1) * abs(grad * etaGain) / etaGain;

    pgrad = (iter < mom_iter ? mom : final_mom) * pgrad - etaGain * grad;

    Gain[i] = gain;
    PrevGradients[i] = pgrad;
    Positions[i] += pgrad * mult;
  }
);

/*
bounds_source GLSL compute shader is designed to find the bounding box of a set of 2D points with added padding.

  The shader uses shared memory to perform a parallel reduction to find the min and max bounds.

   Parameters
    Positions[] : The input array of 2D coordinates.
    Bounds[] : The output array that will contain the minimum and maximum bounds of the points.
    num_points : The number of points in the Positions array  
    padding : The padding factor to be added to the bounds.
*/
const char* bounds_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { vec2 Bounds[]; };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  shared vec2 min_reduction[64];
  shared vec2 max_reduction[64];

  uniform uint num_points;
  uniform float padding;

  void main() {
    uint lid = gl_LocalInvocationIndex.x;
    uint groupSize = gl_WorkGroupSize.x;

    vec2 minBound = vec2(1e38);//1.0 / 0.0); // inf
    vec2 maxBound = vec2(-1e38);//-1.0 / 0.0); // -inf

    // Find min and max bounds by striding (by groupSize) across the positions in each thread.
    for (uint i = lid; i < num_points; i += groupSize)
    {
      vec2 pos = Positions[i];

      minBound = min(pos, minBound);
      maxBound = max(pos, maxBound);
    }

    // Reduce bounds
    // Initialize the reduction arrays from the top 64 thread min and max values
    if (lid >= 64) {
      min_reduction[lid - 64] = minBound;
      max_reduction[lid - 64] = maxBound;
    }
    // Barrier synchronize the threads in the work group
    // and parallel reduce the min and max values in shared memory 
    barrier();
    if (lid < 64) {
      min_reduction[lid] = min(minBound, min_reduction[lid]);
      max_reduction[lid] = max(maxBound, max_reduction[lid]);
    }
    barrier();
    if (lid < 32) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 32]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 32]);
    }
    barrier();
    if (lid < 16) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 16]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 16]);
    }
    barrier();
    if (lid < 8) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 8]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 8]);
    }
    barrier();
    if (lid < 4) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 4]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 4]);
    }
    barrier();
    if (lid < 2) {
      min_reduction[lid] = min(min_reduction[lid], min_reduction[lid + 2]);
      max_reduction[lid] = max(max_reduction[lid], max_reduction[lid + 2]);
    }
    barrier();
    if (lid == 0) {
      minBound = min(min_reduction[0], min_reduction[1]);
      maxBound = max(max_reduction[0], max_reduction[1]);

      vec2 padding = (maxBound - minBound) * padding * 0.5;

      minBound -= padding;
      maxBound += padding;

      Bounds[0] = minBound;
      Bounds[1] = maxBound;
    }
  }
);

// The embedding is recentered and (if enabled) scaled to fit within the bounds of the data.
// In practice scaling is used with higher exageration factors. 
const char* center_and_scale_source = GLSL(430,
  layout(std430, binding = 0) buffer Pos{ vec2 Positions[]; };
  layout(std430, binding = 1) buffer BoundsInterface { vec2 Bounds[]; };
  layout(local_size_x = 128, local_size_y = 1, local_size_z = 1) in;

  uniform uint num_points;
  uniform bool scale;
  uniform float diameter;

  void main() {
    uint workGroupID = gl_WorkGroupID.y * gl_NumWorkGroups.x + gl_WorkGroupID.x;
    uint i = workGroupID * gl_WorkGroupSize.x + gl_LocalInvocationID.x;

    if (i >= num_points)
      return;

    vec2 center = (Bounds[0] + Bounds[1]) * 0.5;

    vec2 pos = Positions[i];

    if (scale)
    {
      float range = Bounds[1].x - Bounds[0].x;

      if (range < diameter) //  || range.y < diameter
      {
        float scale_factor = diameter / range;
        pos -= center;
        pos *= scale_factor;
      }
    }
    else
    {
      pos -= center;
    }

    Positions[i] = pos;
  }
);
