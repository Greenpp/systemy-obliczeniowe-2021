import matplotlib.pyplot as plt
import numpy as np
import pyopencl as cl


def create_env():
    context = cl.create_some_context(interactive=False)
    queue = cl.CommandQueue(context)
    prog = cl.Program(
        context, open('./systemy_bliczeniowe/lab6/kernel.cl').read()
    ).build()

    return context, queue, prog


if __name__ == '__main__':
    context, queue, prog = create_env()

    im_src = plt.imread('./systemy_bliczeniowe/lab6/Lenna.png').astype(np.float32)
    im_dst = np.empty_like(im_src, dtype=np.float32)

    im_shape = im_src.shape[0:2]
    src_buff = cl.image_from_array(context, im_src, mode='r', num_channels=3)
    dst_buff = cl.image_from_array(context, im_dst, mode='w', num_channels=3)

    kernel = np.array(
        [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1],
        ]
    ).astype(np.float32)
    kernel_buff = cl.Buffer(
        context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=kernel
    )

    global_size = im_shape[::-1]
    local_size = None
    prog.conv(queue, global_size, local_size, src_buff, dst_buff, kernel_buff)
    cl.enqueue_copy(
        queue,
        dest=im_dst,
        src=dst_buff,
        is_blocking=True,
        origin=(0, 0),
        region=im_shape[::-1],
    )
    fig, (ax1, ax2) = plt.subplots(1, 2)
    ax1.imshow(im_src)
    ax2.imshow(im_dst)
    plt.savefig("./systemy_bliczeniowe/lab6/imageConv.png", bbox_inches='tight')
