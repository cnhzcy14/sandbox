#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <errno.h>
#include <stdint.h>

// DMA-BUF heap相关定义（如果系统头文件没有）
#ifndef DMA_HEAP_IOC_MAGIC
#define DMA_HEAP_IOC_MAGIC 'H'

struct dma_heap_allocation_data
{
    uint64_t len;
    uint32_t fd;
    uint32_t fd_flags;
    uint64_t heap_flags;
};

#define DMA_HEAP_IOCTL_ALLOC _IOWR(DMA_HEAP_IOC_MAGIC, 0x0, struct dma_heap_allocation_data)
#endif

// 尝试多个可能的heap路径
const char *heap_paths[] = {
    "/dev/dma_heap/system",
    "/dev/dma_heap/linux,cma",
    "/dev/dma_heap/reserved",
    "/dev/dma_heap/system-uncached",
    NULL};

void print_usage(const char *prog)
{
    printf("Usage: %s [size_in_kb]\n", prog);
    printf("Example: %s 4096  (allocate 4MB)\n", prog);
    printf("\nAvailable DMA heaps:\n");

    for (int i = 0; heap_paths[i] != NULL; i++)
    {
        if (access(heap_paths[i], F_OK) == 0)
        {
            printf("  ✓ %s\n", heap_paths[i]);
        }
        else
        {
            printf("  ✗ %s (not available)\n", heap_paths[i]);
        }
    }
}

int allocate_dmabuf(const char *heap_path, size_t size, int *out_fd)
{
    int heap_fd;
    struct dma_heap_allocation_data heap_data;
    int ret;

    printf("\n=== Allocating from %s ===\n", heap_path);

    // 打开heap设备
    heap_fd = open(heap_path, O_RDWR | O_CLOEXEC);
    if (heap_fd < 0)
    {
        fprintf(stderr, "Failed to open %s: %s\n", heap_path, strerror(errno));
        return -1;
    }

    // 准备分配请求
    memset(&heap_data, 0, sizeof(heap_data));
    heap_data.len = size;
    heap_data.fd_flags = O_RDWR | O_CLOEXEC;
    heap_data.heap_flags = 0;

    // 执行分配
    ret = ioctl(heap_fd, DMA_HEAP_IOCTL_ALLOC, &heap_data);
    if (ret < 0)
    {
        fprintf(stderr, "ioctl DMA_HEAP_IOCTL_ALLOC failed: %s\n", strerror(errno));
        close(heap_fd);
        return -1;
    }

    printf("✓ Successfully allocated %zu bytes\n", size);
    printf("✓ DMA-BUF fd: %d\n", heap_data.fd);

    close(heap_fd);
    *out_fd = heap_data.fd;
    return 0;
}

int test_memory_access(int dmabuf_fd, size_t size)
{
    void *addr;

    printf("\n=== Testing Memory Access ===\n");

    // 映射到用户空间
    addr = mmap(NULL, size, PROT_READ | PROT_WRITE, MAP_SHARED, dmabuf_fd, 0);
    if (addr == MAP_FAILED)
    {
        fprintf(stderr, "mmap failed: %s\n", strerror(errno));
        return -1;
    }

    printf("✓ Mapped to address: %p\n", addr);

    // 写入测试数据
    printf("Writing test pattern...\n");
    uint32_t *ptr = (uint32_t *)addr;
    size_t num_words = size / sizeof(uint32_t);

    for (size_t i = 0; i < num_words && i < 1024; i++)
    {
        ptr[i] = 0xDEADBEEF + i;
    }

    // 读取并验证
    printf("Verifying test pattern...\n");
    int errors = 0;
    for (size_t i = 0; i < num_words && i < 1024; i++)
    {
        if (ptr[i] != 0xDEADBEEF + i)
        {
            fprintf(stderr, "Verify error at offset %zu: expected 0x%x, got 0x%x\n",
                    i, 0xDEADBEEF + (uint32_t)i, ptr[i]);
            errors++;
            if (errors >= 10)
                break;
        }
    }

    if (errors == 0)
    {
        printf("✓ Memory verification passed!\n");
    }
    else
    {
        printf("✗ Memory verification failed with %d errors\n", errors);
    }

    // 显示前几个值
    printf("\nFirst 8 words:\n");
    for (int i = 0; i < 8 && i < num_words; i++)
    {
        printf("  [%d] = 0x%08x\n", i, ptr[i]);
    }

    // 解除映射
    munmap(addr, size);
    printf("✓ Unmapped memory\n");

    return errors == 0 ? 0 : -1;
}

void get_buffer_info(int dmabuf_fd)
{
    off_t buffer_size;

    printf("\n=== Buffer Information ===\n");

    // 获取buffer大小
    buffer_size = lseek(dmabuf_fd, 0, SEEK_END);
    if (buffer_size >= 0)
    {
        printf("Buffer size: %ld bytes (%.2f MB)\n",
               buffer_size, buffer_size / (1024.0 * 1024.0));
        lseek(dmabuf_fd, 0, SEEK_SET);
    }

    // 获取fd标志
    int flags = fcntl(dmabuf_fd, F_GETFL);
    if (flags >= 0)
    {
        printf("FD flags: 0x%x ", flags);
        if (flags & O_RDWR)
            printf("(O_RDWR) ");
        if (flags & O_RDONLY)
            printf("(O_RDONLY) ");
        if (flags & O_WRONLY)
            printf("(O_WRONLY) ");
        printf("\n");
    }
}

int main(int argc, char *argv[])
{
    size_t size_kb = 4096; // 默认4MB
    size_t size;
    int dmabuf_fd = -1;
    int ret;
    const char *selected_heap = NULL;

    printf("DMA-BUF Heap Example\n");
    printf("====================\n");

    // 解析命令行参数
    if (argc > 1)
    {
        if (strcmp(argv[1], "-h") == 0 || strcmp(argv[1], "--help") == 0)
        {
            print_usage(argv[0]);
            return 0;
        }
        size_kb = atoi(argv[1]);
        if (size_kb == 0)
        {
            fprintf(stderr, "Invalid size: %s\n", argv[1]);
            print_usage(argv[0]);
            return 1;
        }
    }

    size = size_kb * 1024;
    printf("Requested allocation size: %zu KB (%.2f MB)\n",
           size_kb, size / (1024.0 * 1024.0));

    // 尝试找到可用的heap
    for (int i = 0; heap_paths[i] != NULL; i++)
    {
        if (access(heap_paths[i], F_OK) == 0)
        {
            selected_heap = heap_paths[i];
            break;
        }
    }

    if (selected_heap == NULL)
    {
        fprintf(stderr, "\n✗ ERROR: No DMA heap devices found!\n");
        fprintf(stderr, "\nPossible reasons:\n");
        fprintf(stderr, "1. Kernel doesn't support DMA-BUF heaps (need Linux 5.6+)\n");
        fprintf(stderr, "2. CONFIG_DMABUF_HEAPS is not enabled in kernel\n");
        fprintf(stderr, "3. No heap drivers are loaded\n");
        fprintf(stderr, "\nTo check: dmesg | grep -i dma_heap\n");
        fprintf(stderr, "To check kernel config: grep DMABUF_HEAPS /boot/config-$(uname -r)\n");
        return 1;
    }

    // 分配内存
    ret = allocate_dmabuf(selected_heap, size, &dmabuf_fd);
    if (ret < 0)
    {
        fprintf(stderr, "\n✗ Allocation failed\n");
        fprintf(stderr, "\nTroubleshooting:\n");
        fprintf(stderr, "1. Try a smaller size\n");
        fprintf(stderr, "2. Check available memory: free -h\n");
        fprintf(stderr, "3. Check CMA: cat /proc/meminfo | grep Cma\n");
        fprintf(stderr, "4. Run with sufficient permissions (may need sudo)\n");
        return 1;
    }

    // 获取buffer信息
    get_buffer_info(dmabuf_fd);

    // 测试内存访问
    ret = test_memory_access(dmabuf_fd, size);

    // 清理
    printf("\n=== Cleanup ===\n");
    close(dmabuf_fd);
    printf("✓ Closed DMA-BUF fd\n");

    if (ret == 0)
    {
        printf("\n✓✓✓ All tests passed! ✓✓✓\n");
    }
    else
    {
        printf("\n✗✗✗ Some tests failed ✗✗✗\n");
    }

    return ret;
}
