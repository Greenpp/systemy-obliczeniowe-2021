__constant sampler_t sampler =
    CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
__kernel void conv(read_only image2d_t src_image,
                   write_only image2d_t dst_image,
                   __global float *conv_kernel) {
  size_t global_size_0 = get_global_size(0);
  size_t global_size_1 = get_global_size(1);
  size_t global_id_0 = get_global_id(0);
  size_t global_id_1 = get_global_id(1);
  int2 image_dim = get_image_dim(src_image);

  float width_step = (float)image_dim.x / ((float)global_size_0);
  float height_step = (float)image_dim.y / ((float)global_size_1);
  int2 coord = (int2)(width_step * global_id_0, height_step * global_id_1);

  float4 new_pixel_val = 0.0;
  int conv_kernel_idx = 0;
  for (int i = -1; i <= 1; i++) {
    for (int j = -1; j <= 1; j++) {
      int2 conv_coord = (int2)(coord.x + i, coord.y + j);
      float4 conv_px = read_imagef(src_image, sampler, conv_coord);

      new_pixel_val += conv_px * *(conv_kernel + conv_kernel_idx);

      conv_kernel_idx++;
    }
  }
  write_imagef(dst_image, coord, new_pixel_val);
}