#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdint.h>

// DMA-BUF heap 定义
#define DMA_HEAP_IOC_MAGIC 'H'

struct dma_heap_allocation_data
{
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, \
                                   struct dma_heap_allocation_data)

int main()
{
    int heap_fd, dmabuf_fd;
    struct dma_heap_allocation_data alloc_data;
    void *addr;
    size_t size = 4 * 1024 * 1024; // 4MB

    printf("用户态申请 DMA-BUF heap 示例\n");
    printf("==============================\n\n");

    // 1. 打开 heap 设备
    printf("步骤 1: 打开 /dev/dma_heap/linux,cma\n");
    heap_fd = open("/dev/dma_heap/linux,cma", O_RDWR | O_CLOEXEC);
    if (heap_fd < 0)
    {
        // 如果 CMA 不可用，尝试 system heap
        printf("   CMA 不可用，尝试 system heap...\n");
        heap_fd = open("/dev/dma_heap/system", O_RDWR | O_CLOEXEC);
        if (heap_fd < 0)
        {
            perror("   打开 heap 设备失败");
            return 1;
        }
    }
    printf("   ✓ heap_fd = %d\n\n", heap_fd);

    // 2. 准备分配请求
    printf("步骤 2: 准备分配请求 (4MB)\n");
    memset(&alloc_data, 0, sizeof(alloc_data));
    alloc_data.len = size;
    alloc_data.fd_flags = O_RDWR | O_CLOEXEC;
    alloc_data.heap_flags = 0;
    printf("   ✓ size = %zu bytes\n\n", size);

    // 3. 执行分配
    printf("步骤 3: 调用 ioctl(DMA_HEAP_IOCTL_ALLOC)\n");
    if (ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &alloc_data) < 0)
    {
        perror("   ioctl 失败");
        close(heap_fd);
        return 1;
    }
    dmabuf_fd = alloc_data.fd;
    printf("   ✓ 分配成功! dmabuf_fd = %d\n\n", dmabuf_fd);

    // heap_fd 不再需要
    close(heap_fd);

    // 4. 映射到用户空间
    printf("步骤 4: mmap() 映射到用户空间\n");
    addr = mmap(NULL, size, PROT_READ | PROT_WRITE,
                MAP_SHARED, dmabuf_fd, 0);
    if (addr == MAP_FAILED)
    {
        perror("   mmap 失败");
        close(dmabuf_fd);
        return 1;
    }
    printf("   ✓ 映射成功! addr = %p\n\n", addr);

    // 5. 使用内存
    printf("步骤 5: 使用内存 (写入测试)\n");
    uint32_t *ptr = (uint32_t *)addr;
    for (int i = 0; i < 10; i++)
    {
        ptr[i] = 0xDEADBEEF + i;
    }
    printf("   ✓ 写入完成\n");

    // 验证
    printf("   验证前 10 个 uint32_t:\n");
    for (int i = 0; i < 10; i++)
    {
        printf("   [%d] = 0x%08x\n", i, ptr[i]);
    }
    printf("\n");

    // 6. 清理
    printf("步骤 6: 清理资源\n");
    munmap(addr, size);
    printf("   ✓ munmap 完成\n");
    close(dmabuf_fd);
    printf("   ✓ close(dmabuf_fd) 完成\n\n");

    printf("==============================\n");
    printf("✓ 用户态申请 DMA-BUF 成功!\n");
    printf("==============================\n");

    return 0;
}