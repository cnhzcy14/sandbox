/*
 * Demo application for DMA buffer sharing between V4L2 and DRM
 * Tomasz Stanislawski <t.stanisl...@samsung.com>
 *
 * Copyright 2012 Samsung Electronics Co., Ltd.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 */

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <pthread.h>

#include <drm.h>
#include <drm_mode.h>

//#include <linux/videodev2.h>
#include "videodev2.h"

#include <xf86drm.h>
#include <xf86drmMode.h>

#define ERRSTR strerror(errno)

#define BYE_ON(cond, ...) \
do { \
	if (cond) { \
		int errsv = errno; \
		fprintf(stderr, "ERROR(%s:%d) : ", \
			__FILE__, __LINE__); \
		errno = errsv; \
		fprintf(stderr,  __VA_ARGS__); \
		abort(); \
	} \
} while(0)

static inline int warn(const char *file, int line, const char *fmt, ...)
{
	int errsv = errno;
	va_list va;
	va_start(va, fmt);
	fprintf(stderr, "WARN(%s:%d): ", file, line);
	vfprintf(stderr, fmt, va);
	va_end(va);
	errno = errsv;
	return 1;
}

#define WARN_ON(cond, ...) \
	((cond) ? warn(__FILE__, __LINE__, __VA_ARGS__) : 0)

struct crtc {
	drmModeCrtc *crtc;
	drmModeObjectProperties *props;
	drmModePropertyRes **props_info;
};

struct encoder {
	drmModeEncoder *encoder;
};

struct connector {
	drmModeConnector *connector;
	drmModeObjectProperties *props;
	drmModePropertyRes **props_info;
};

struct plane {
        drmModePlane *plane;
        drmModeObjectProperties *props;
        drmModePropertyRes **props_info;
};

struct resources {
	int fd;
	drmModeResPtr res;
	drmModePlaneResPtr plane_res;
	struct crtc *crtcs;
	struct encoder *encoders;
	struct connector *connectors;
	struct plane *planes;
};

struct setup {
	char module[32];
	uint32_t conId;
	uint32_t crtcId;
	int crtcIdx;
	uint32_t planeId;
	char video[32];
	unsigned int w, h;
	unsigned int use_wh : 1;
	unsigned int in_fourcc;
	unsigned int out_fourcc;
	unsigned int buffer_count;
	unsigned int times;
	unsigned int use_crop : 1;
	unsigned int use_compose : 1;
	struct v4l2_rect crop;
	struct v4l2_rect compose;
};

struct buffer {
	unsigned int bo_handle;
	unsigned int fb_handle;
	int dbuf_fd;
	int queued_in_v4l2;
	int v4l2_out_fence;
	int drm_out_fence;
};

struct stream {
	int drm_next;
	struct buffer *buffer;
} stream;

static void usage(char *name)
{
	fprintf(stderr, "usage: %s [-Moisth]\n", name);
	fprintf(stderr, "\t-M <drm-module>\tset DRM module\n");
	fprintf(stderr, "\t-o <connector_id>:<crtc_id>\tchoose a connector/crtc\n");
	fprintf(stderr, "\t-i <video-node>\tset video node like /dev/video*\n");
	fprintf(stderr, "\t-S <width,height>\tset input resolution\n");
	fprintf(stderr, "\t-f <fourcc>\tset input format using 4cc\n");
	fprintf(stderr, "\t-F <fourcc>\tset output format using 4cc\n");
	fprintf(stderr, "\t-s <width,height>@<left,top>\tset crop area\n");
	fprintf(stderr, "\t-t <width,height>@<left,top>\tset compose area\n");
	fprintf(stderr, "\t-b buffer_count\tset number of buffers\n");
	fprintf(stderr, "\t-n <times>\tnumber of iterations in the pipeline\n");
	fprintf(stderr, "\t-h\tshow this help\n");
	fprintf(stderr, "\n\tDefault is to dump all info.\n");
}

static inline int parse_rect(char *s, struct v4l2_rect *r)
{
	return sscanf(s, "%d,%d@%d,%d", &r->width, &r->height,
		&r->top, &r->left) != 4;
}

static int parse_args(int argc, char *argv[], struct setup *s)
{
	if (argc <= 1)
		usage(argv[0]);

	int c, ret;
	memset(s, 0, sizeof(*s));

	while ((c = getopt(argc, argv, "M:o:i:S:f:F:s:t:b:n:h")) != -1) {
		switch (c) {
		case 'M':
			strncpy(s->module, optarg, 31);
			break;
		case 'o':
			ret = sscanf(optarg, "%u:%u", &s->conId, &s->crtcId);
			if (WARN_ON(ret != 2, "incorrect con/ctrc description\n"))
				return -1;
			break;
		case 'i':
			strncpy(s->video, optarg, 31);
			break;
		case 'S':
			ret = sscanf(optarg, "%u,%u", &s->w, &s->h);
			if (WARN_ON(ret != 2, "incorrect input size\n"))
				return -1;
			s->use_wh = 1;
			break;
		case 'f':
			if (WARN_ON(strlen(optarg) != 4, "invalid fourcc\n"))
				return -1;
			s->in_fourcc = ((unsigned)optarg[0] << 0) |
				((unsigned)optarg[1] << 8) |
				((unsigned)optarg[2] << 16) |
				((unsigned)optarg[3] << 24);
			break;
		case 'F':
			if (WARN_ON(strlen(optarg) != 4, "invalid fourcc\n"))
				return -1;
			s->out_fourcc = ((unsigned)optarg[0] << 0) |
				((unsigned)optarg[1] << 8) |
				((unsigned)optarg[2] << 16) |
				((unsigned)optarg[3] << 24);
			break;
		case 's':
			ret = parse_rect(optarg, &s->crop);
			if (WARN_ON(ret, "incorrect crop area\n"))
				return -1;
			s->use_crop = 1;
			break;
		case 't':
			ret = parse_rect(optarg, &s->compose);
			if (WARN_ON(ret, "incorrect compose area\n"))
				return -1;
			s->use_compose = 1;
			break;
		case 'b':
			ret = sscanf(optarg, "%u", &s->buffer_count);
			if (WARN_ON(ret != 1, "incorrect buffer count\n"))
				return -1;
			break;
		case 'n':
			ret = sscanf(optarg, "%u", &s->times);
			if (WARN_ON(ret != 1, "incorrect number of times to run\n"))
				return -1;
			break;
		case '?':
		case 'h':
			usage(argv[0]);
			return -1;
		}
	}

	return 0;
}

static int buffer_create(struct buffer *b, int drmfd, struct setup *s,
	uint64_t size, uint32_t pitch)
{
	struct drm_mode_create_dumb gem;
	struct drm_mode_destroy_dumb gem_destroy;
	int ret;

	memset(&gem, 0, sizeof gem);
	gem.width = s->w;
	gem.height = s->h;
	gem.bpp = 32;
	gem.size = size;
	ret = ioctl(drmfd, DRM_IOCTL_MODE_CREATE_DUMB, &gem);
	if (WARN_ON(ret, "CREATE_DUMB failed: %s\n", ERRSTR))
		return -1;
	printf("bo %u %ux%u bpp %u size %lu (%lu)\n", gem.handle, gem.width, gem.height, gem.bpp, (long)gem.size, (long)size);
	b->bo_handle = gem.handle;

	struct drm_prime_handle prime;
	memset(&prime, 0, sizeof prime);
	prime.handle = b->bo_handle;

	ret = ioctl(drmfd, DRM_IOCTL_PRIME_HANDLE_TO_FD, &prime);
	if (WARN_ON(ret, "PRIME_HANDLE_TO_FD failed: %s\n", ERRSTR))
		goto fail_gem;
	printf("dbuf_fd = %d\n", prime.fd);
	b->dbuf_fd = prime.fd;

	uint32_t offsets[4] = { 0 };
	uint32_t pitches[4] = { pitch };
	uint32_t bo_handles[4] = { b->bo_handle };
	unsigned int fourcc = s->out_fourcc;
	if (!fourcc)
		fourcc = s->in_fourcc;

	fprintf(stderr, "FB fourcc %c%c%c%c\n",
		fourcc,
		fourcc >> 8,
		fourcc >> 16,
		fourcc >> 24);

	ret = drmModeAddFB2(drmfd, s->w, s->h, fourcc, bo_handles,
		pitches, offsets, &b->fb_handle, 0);
	if (WARN_ON(ret, "drmModeAddFB2 failed: %s\n", ERRSTR))
		goto fail_prime;

	return 0;

fail_prime:
	close(b->dbuf_fd);

fail_gem:
	memset(&gem_destroy, 0, sizeof gem_destroy);
	gem_destroy.handle = b->bo_handle,
	ret = ioctl(drmfd, DRM_IOCTL_MODE_DESTROY_DUMB, &gem_destroy);
	WARN_ON(ret, "DESTROY_DUMB failed: %s\n", ERRSTR);

	return -1;
}

static int find_crtc(struct resources *res, struct setup *s)
{
	int ret = -1;
	int i;

	if (!s->conId) {
		fprintf(stderr,
			"No connector ID specified.  Choosing default from list:\n");

		for (i = 0; i < res->res->count_connectors; i++) {
			drmModeConnector *con = res->connectors[i].connector;
			drmModeEncoder *enc = NULL;
			drmModeCrtc *crtc = NULL;

			if (con->encoder_id) {
				enc = drmModeGetEncoder(res->fd, con->encoder_id);
				if (enc->crtc_id) {
					crtc = drmModeGetCrtc(res->fd, enc->crtc_id);
				}
			}

			if (!s->conId && crtc) {
				s->conId = con->connector_id;
				s->crtcId = crtc->crtc_id;
			}

			printf("Connector %d (crtc %d): type %d, %dx%d%s\n",
			       con->connector_id,
			       crtc ? crtc->crtc_id : 0,
			       con->connector_type,
			       crtc ? crtc->width : 0,
			       crtc ? crtc->height : 0,
			       (s->conId == con->connector_id ?
				" (chosen)" : ""));
		}

		if (!s->conId) {
			fprintf(stderr,
				"No suitable enabled connector found.\n");
			exit(1);
		}
	}

	s->crtcIdx = -1;

	for (i = 0; i < res->res->count_crtcs; ++i) {
		if (s->crtcId == res->res->crtcs[i]) {
			s->crtcIdx = i;
			break;
		}
	}

	if (WARN_ON(s->crtcIdx == -1, "drm: CRTC %u not found\n", s->crtcId))
		return ret;

	if (WARN_ON(res->res->count_connectors <= 0, "drm: no connectors\n"))
		return ret;

	drmModeConnector *c = NULL;
	for (i = 0; i < res->res->count_connectors; ++i) {
		if (s->conId == res->connectors[i].connector->connector_id) {
			c = res->connectors[i].connector;
			break;
		}
	}
	if (WARN_ON(!c, "drmModeGetConnector failed: %s\n", ERRSTR))
		return ret;

	if (WARN_ON(!c->count_modes, "connector supports no mode\n"))
		return ret;

	if (!s->use_compose) {
		drmModeCrtc *crtc = NULL;
		for (i = 0; i < res->res->count_crtcs; ++i) {
			if (s->crtcId == res->crtcs[i].crtc->crtc_id) {
				crtc = res->crtcs[i].crtc;
				break;
			}
		}
		s->compose.left = crtc->x;
		s->compose.top = crtc->y;
		s->compose.width = crtc->width;
		s->compose.height = crtc->height;
		drmModeFreeCrtc(crtc);
	}

	return 0;
}

static void free_resources(struct resources *res)
{
	int i;

	if (!res)
		return;

#define free_resource(_res, __res, type, Type)					\
	do {									\
		if (!(_res)->type##s)						\
			break;							\
		for (i = 0; i < (int)(_res)->__res->count_##type##s; ++i) {	\
			if (!(_res)->type##s[i].type)				\
				break;						\
			drmModeFree##Type((_res)->type##s[i].type);		\
		}								\
		free((_res)->type##s);						\
	} while (0)

#define free_properties(_res, __res, type)					\
	do {									\
		for (i = 0; i < (int)(_res)->__res->count_##type##s; ++i) {	\
			drmModeFreeObjectProperties(res->type##s[i].props);	\
			free(res->type##s[i].props_info);			\
		}								\
	} while (0)

	if (res->res) {
		free_properties(res, res, crtc);
		free_properties(res, res, connector);

		free_resource(res, res, crtc, Crtc);
		free_resource(res, res, encoder, Encoder);
		free_resource(res, res, connector, Connector);

		drmModeFreeResources(res->res);
	}

	if (res->plane_res) {
		free_properties(res, plane_res, plane);

		free_resource(res, plane_res, plane, Plane);

		drmModeFreePlaneResources(res->plane_res);
	}
}

static int get_drm_resources(struct resources *res)
{
	int i;

	res->res = drmModeGetResources(res->fd);
	if (WARN_ON(!res->res, "drmModeGetResources failed: %s\n", ERRSTR))
		return -1;

	if (WARN_ON(res->res->count_crtcs <= 0, "drm: no crts\n"))
		return -1;

	res->crtcs = calloc(res->res->count_crtcs, sizeof(*res->crtcs));
	if (WARN_ON(!res->crtcs, "calloc failed: %s\n", ERRSTR))
		return -1;

	res->connectors = calloc(res->res->count_connectors, sizeof(*res->connectors));
	if (WARN_ON(!res->connectors, "calloc failed: %s\n", ERRSTR))
		return -1;

	res->encoders = calloc(res->res->count_encoders, sizeof(*res->encoders));
	if (WARN_ON(!res->encoders, "calloc failed: %s\n", ERRSTR))
		return -1;

#define get_resource(_res, __res, type, Type)                                   \
        do {                                                                    \
                for (i = 0; i < (int)(_res)->__res->count_##type##s; ++i) {     \
                        (_res)->type##s[i].type =                               \
                                drmModeGet##Type((_res)->fd, (_res)->__res->type##s[i]); \
                        if (!(_res)->type##s[i].type)                           \
                                fprintf(stderr, "could not get %s %i: %s\n",    \
                                        #type, (_res)->__res->type##s[i],       \
                                        strerror(errno));                       \
                }                                                               \
        } while (0)

	get_resource(res, res, crtc, Crtc);
	get_resource(res, res, connector, Connector);
	get_resource(res, res, encoder, Encoder);

#define get_properties(_res, __res, type, Type)                                 \
        do {                                                                    \
                for (i = 0; i < (int)(_res)->__res->count_##type##s; ++i) {     \
                        struct type *obj = &res->type##s[i];                    \
                        unsigned int j;                                         \
                        obj->props =                                            \
                                drmModeObjectGetProperties(res->fd, obj->type->type##_id, \
                                                           DRM_MODE_OBJECT_##Type); \
                        if (!obj->props) {                                      \
                                fprintf(stderr,                                 \
                                        "could not get %s %i properties: %s\n", \
                                        #type, obj->type->type##_id,            \
                                        strerror(errno));                       \
                                continue;                                       \
                        }                                                       \
                        obj->props_info = calloc(obj->props->count_props,       \
                                                 sizeof(*obj->props_info));     \
                        if (!obj->props_info)                                   \
                                continue;                                       \
                        for (j = 0; j < obj->props->count_props; ++j)           \
                                obj->props_info[j] =                            \
                                        drmModeGetProperty(res->fd, obj->props->props[j]); \
                }                                                               \
        } while (0)

	get_properties(res, res, crtc, CRTC);
	get_properties(res, res, connector, CONNECTOR);

	res->plane_res = drmModeGetPlaneResources(res->fd);
	if (WARN_ON(!res->plane_res, "drmModeGetPlaneResources failed: %s\n", ERRSTR))
		return -1;

	res->planes = calloc(res->plane_res->count_planes, sizeof(*res->planes));
	if (WARN_ON(!res->planes, "calloc failed: %s\n", ERRSTR))
		return -1;

	get_resource(res, plane_res, plane, Plane);
        get_properties(res, plane_res, plane, PLANE);

	return 0;
}

static int add_plane_property(struct resources *res, drmModeAtomicReq *req,
				uint32_t obj_id, const char *name, uint64_t value)
{
	struct plane *obj = NULL;
	unsigned int i;
	int prop_id = -1;

	for (i = 0; i < (unsigned int) res->plane_res->count_planes ; i++) {
		if (obj_id == res->plane_res->planes[i]) {
			obj = &res->planes[i];
			break;
		}
	}

	if (!obj)
		return -EINVAL;

	for (i = 0 ; i < obj->props->count_props ; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			prop_id = obj->props_info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0)
		return -EINVAL;

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
}

static int add_crtc_property(struct resources *res, drmModeAtomicReq *req,
				uint32_t obj_id, const char *name, uint64_t value)
{
	struct crtc *obj = NULL;
	unsigned int i;
	int prop_id = -1;

	for (i = 0; i < (unsigned int) res->res->count_crtcs ; i++) {
		if (obj_id == res->res->crtcs[i]) {
			obj = &res->crtcs[i];
			break;
		}
	}

	if (!obj)
		return -EINVAL;

	for (i = 0 ; i < obj->props->count_props ; i++) {
		if (strcmp(obj->props_info[i]->name, name) == 0) {
			prop_id = obj->props_info[i]->prop_id;
			break;
		}
	}

	if (prop_id < 0)
		return -EINVAL;

	return drmModeAtomicAddProperty(req, obj_id, prop_id, value);
}

int set_plane(struct resources *res, struct setup *s, struct buffer *buffer)
{
	drmModeAtomicReq *req;
	uint64_t fence = 0;
	int ret;

	req = drmModeAtomicAlloc();

	add_plane_property(res, req, s->planeId, "FB_ID", buffer->fb_handle);
	add_plane_property(res, req, s->planeId, "CRTC_ID", s->crtcId);
	add_plane_property(res, req, s->planeId, "SRC_X", 0);
	add_plane_property(res, req, s->planeId, "SRC_Y", 0);
	add_plane_property(res, req, s->planeId, "SRC_W", s->w << 16);
	add_plane_property(res, req, s->planeId, "SRC_H", s->h << 16);
	add_plane_property(res, req, s->planeId, "CRTC_X", s->compose.left);
	add_plane_property(res, req, s->planeId, "CRTC_Y", s->compose.top);
	add_plane_property(res, req, s->planeId, "CRTC_W", s->compose.width);
	add_plane_property(res, req, s->planeId, "CRTC_H", s->compose.height);
	add_plane_property(res, req, s->planeId, "IN_FENCE_FD", buffer->v4l2_out_fence);

	ret = add_crtc_property(res, req, s->crtcId, "OUT_FENCE_PTR", (uint64_t)(void *)&fence);
	BYE_ON(ret < 0, "add out fence failed: (%d)\n", ret);

	ret = drmModeAtomicCommit(res->fd, req, DRM_MODE_PAGE_FLIP_EVENT, NULL);
	BYE_ON(ret < 0, "atomic commit failed: (%d)\n", ret);

	buffer->drm_out_fence = fence;
	close(buffer->v4l2_out_fence);
	buffer->v4l2_out_fence = -1;

	drmModeAtomicFree(req);

	return ret;
}

static int find_plane(struct resources *res, struct setup *s)
{
	drmModePlanePtr plane;
	unsigned int i;
	unsigned int j;

	for (i = 0; i < res->plane_res->count_planes; ++i) {
		plane = res->planes[i].plane;
		for (j = 0; j < plane->count_formats; ++j) {
			if (plane->formats[j] == s->out_fourcc)
				break;
		}

		if (j == plane->count_formats) {
			continue;
		}

		s->planeId = plane->plane_id;
		break;
	}

	if (i == res->plane_res->count_planes)
		return -1;

	return 0;
}

int v4lfd;

int next_drm_buffer(int cnt)
{
	return (stream.drm_next + 1) % cnt;
}

void v4l2_queue_buffer(struct buffer *buffer, int index) {
	struct v4l2_buffer buf;
	int ret;

	if (buffer[index].queued_in_v4l2)
		return;

	if (buffer[index].drm_out_fence == -1)
		return;

	memset(&buf, 0, sizeof buf);
	buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	buf.memory = V4L2_MEMORY_DMABUF;
	buf.index = index;
	buf.flags = V4L2_BUF_FLAG_OUT_FENCE;
	if (buffer[index].drm_out_fence >= 0) {
		buf.fence_fd = buffer[index].drm_out_fence;
		buf.flags |= V4L2_BUF_FLAG_IN_FENCE;
	}
	buf.m.fd = buffer[index].dbuf_fd;

	ret = ioctl(v4lfd, VIDIOC_QBUF, &buf);
	BYE_ON(ret, "VIDIOC_QBUF(index = %d) failed: %s\n", index, ERRSTR);
	buffer[index].v4l2_out_fence = buf.fence_fd;
	close(buffer[index].drm_out_fence);
	buffer[index].drm_out_fence = -1;
}

void *v4l2_thread(void *arg)
{
	struct buffer *buffer = arg;
	struct v4l2_buffer buf;
	struct pollfd fds[] = {
		{ .fd = v4lfd, .events = POLLIN },
	};
	int ret;

	while ((ret = poll(fds, 1, 5000)) > 0) {
		memset(&buf, 0, sizeof buf);

		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_DMABUF;
		ret = ioctl(v4lfd, VIDIOC_DQBUF, &buf);
		BYE_ON(ret, "VIDIOC_DQBUF failed: %s\n", ERRSTR);

		buffer[buf.index].queued_in_v4l2 = 0;

		v4l2_queue_buffer(buffer, buf.index);

	}
	pthread_exit(NULL);
}

int main(int argc, char *argv[])
{
	struct setup s;
	int ret;

	ret = parse_args(argc, argv, &s);
	BYE_ON(ret, "failed to parse arguments\n");
	BYE_ON(s.module[0] == 0, "DRM module is missing\n");
	BYE_ON(s.video[0] == 0, "video node is missing\n");

	int drmfd = drmOpen(s.module, NULL);
	BYE_ON(drmfd < 0, "drmOpen(%s) failed: %s\n", s.module, ERRSTR);

	v4lfd = open(s.video, O_RDWR);
	BYE_ON(v4lfd < 0, "failed to open %s: %s\n", s.video, ERRSTR);

	struct v4l2_capability caps;
	memset(&caps, 0, sizeof caps);

	ret = ioctl(v4lfd, VIDIOC_QUERYCAP, &caps);
	BYE_ON(ret, "VIDIOC_QUERYCAP failed: %s\n", ERRSTR);

	/* TODO: add single plane support */
	BYE_ON(~caps.capabilities & V4L2_CAP_VIDEO_CAPTURE,
		"video: singleplanar capture is not supported\n");

	struct v4l2_format fmt;
	memset(&fmt, 0, sizeof fmt);
	fmt.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;

	ret = ioctl(v4lfd, VIDIOC_G_FMT, &fmt);
	BYE_ON(ret < 0, "VIDIOC_G_FMT failed: %s\n", ERRSTR);
	printf("G_FMT(start): width = %u, height = %u, 4cc = %.4s\n",
		fmt.fmt.pix.width, fmt.fmt.pix.height,
		(char*)&fmt.fmt.pix.pixelformat);

	if (s.use_wh) {
		fmt.fmt.pix.width = s.w;
		fmt.fmt.pix.height = s.h;
	}
	if (s.in_fourcc)
		fmt.fmt.pix.pixelformat = s.in_fourcc;

	ret = ioctl(v4lfd, VIDIOC_S_FMT, &fmt);
	BYE_ON(ret < 0, "VIDIOC_S_FMT failed: %s\n", ERRSTR);

	ret = ioctl(v4lfd, VIDIOC_G_FMT, &fmt);
	BYE_ON(ret < 0, "VIDIOC_G_FMT failed: %s\n", ERRSTR);
	printf("G_FMT(final): width = %u, height = %u, 4cc = %.4s\n",
		fmt.fmt.pix.width, fmt.fmt.pix.height,
		(char*)&fmt.fmt.pix.pixelformat);

	struct v4l2_requestbuffers rqbufs;
	memset(&rqbufs, 0, sizeof(rqbufs));
	rqbufs.count = s.buffer_count;
	rqbufs.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	rqbufs.memory = V4L2_MEMORY_DMABUF;

	ret = ioctl(v4lfd, VIDIOC_REQBUFS, &rqbufs);
	BYE_ON(ret < 0, "VIDIOC_REQBUFS failed: %s\n", ERRSTR);
	BYE_ON(rqbufs.count < s.buffer_count, "video node allocated only "
		"%u of %u buffers\n", rqbufs.count, s.buffer_count);

	s.in_fourcc = fmt.fmt.pix.pixelformat;
	s.w = fmt.fmt.pix.width;
	s.h = fmt.fmt.pix.height;

	/* TODO: add support for multiplanar formats */
	struct buffer buffer[s.buffer_count];
	uint32_t size = fmt.fmt.pix.sizeimage;
	uint32_t pitch = fmt.fmt.pix.bytesperline;
	printf("size = %u pitch = %u\n", size, pitch);
	for (unsigned int i = 0; i < s.buffer_count; ++i) {
		ret = buffer_create(&buffer[i], drmfd, &s, size, pitch);
		BYE_ON(ret, "failed to create buffer%d\n", i);
		buffer[i].v4l2_out_fence = -1;
		buffer[i].drm_out_fence = -1;
	}
	printf("buffers ready\n");

	drmSetClientCap(drmfd, DRM_CLIENT_CAP_ATOMIC, 1);

	struct resources resources;
	resources.fd = drmfd;
	ret = get_drm_resources(&resources);
	BYE_ON(ret, "failed to get drm resources\n");

	//XXX convert find_crtc to use resources
	ret = find_crtc(&resources, &s);
	BYE_ON(ret, "failed to find valid mode\n");

	ret = find_plane(&resources, &s);
	BYE_ON(ret, "failed to find compatible plane\n");

	for (unsigned int i = 0; i < s.buffer_count; ++i) {
		struct v4l2_buffer buf;
		memset(&buf, 0, sizeof buf);

		buf.index = i;
		buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
		buf.memory = V4L2_MEMORY_DMABUF;
		buf.m.fd = buffer[i].dbuf_fd;
		buf.flags |= V4L2_BUF_FLAG_OUT_FENCE;
		ret = ioctl(v4lfd, VIDIOC_QBUF, &buf);
		BYE_ON(ret < 0, "VIDIOC_QBUF for buffer %d failed: %s (fd %u)\n",
			buf.index, ERRSTR, buffer[i].dbuf_fd);

		buffer[i].v4l2_out_fence = buf.fence_fd;
		buffer[i].queued_in_v4l2 = 1;
	}

	int type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
	ret = ioctl(v4lfd, VIDIOC_STREAMON, &type);
	BYE_ON(ret < 0, "STREAMON failed: %s\n", ERRSTR);

	unsigned int n = 0;
	pthread_t v4l2_pthread;

	pthread_create(&v4l2_pthread, NULL, v4l2_thread, buffer);

	drmEventContext evctx;
	memset(&evctx, 0, sizeof evctx);
	evctx.version = DRM_EVENT_CONTEXT_VERSION;
	evctx.vblank_handler = NULL;
	evctx.page_flip_handler = NULL;

	ret = set_plane(&resources, &s, &buffer[0]);
	BYE_ON(ret, "failed to set plane: %s\n", ERRSTR);
	stream.drm_next = 1;

	n++;

	struct pollfd fds[] = {
		{ .fd = drmfd, .events = POLLIN },
	};

	while ((ret = poll(fds, 1, 5000)) > 0) {
		drmHandleEvent(drmfd, &evctx);

		// XXX make this NONBLOCKING
		ret = set_plane(&resources, &s, &buffer[stream.drm_next]);
		BYE_ON(ret, "failed to set plane: %s\n", ERRSTR);

		if (++n == s.times)
			break;

		v4l2_queue_buffer(buffer, stream.drm_next);

		stream.drm_next = next_drm_buffer(s.buffer_count);
	}

	//XXX clean up pthreads

	free_resources(&resources);
	return 0;
}
