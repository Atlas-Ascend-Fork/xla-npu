#ifndef STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_
#define STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_

#include "xla/xla_npu/stream_executor/npu_stream.h"

namespace stream_executor {

class NpuStreamFactory {
   public:
    virtual ~NpuStreamFactory() = default;
    virtual std::unique_ptr<NpuStream> CreateStream(StreamExecutor* executor) const = 0;
    static void Register(std::unique_ptr<NpuStreamFactory> factory, int priority);
    static const NpuStreamFactory* GetFactory();
};

class NpuStreamDefaultFactory : public NpuStreamFactory {
   public:
    ~NpuStreamDefaultFactory() override = default;
    std::unique_ptr<NpuStream> CreateStream(StreamExecutor* executor) const override {
        return std::make_unique<NpuStream>(executor);
    }
};

template <class Factory>
class NpuStreamFactoryRegistrar {
   public:
    explicit NpuStreamFactoryRegistrar(int priority) {
        NpuStreamFactory::Register(std::make_unique<Factory>(), priority);
    }
};

}  // namespace stream_executor

#define REGISTER_NPU_STREAM_FACTORY(factory, priority) \
    INTERNAL_REGISTER_NPU_STREAM_FACTORY(factory, priority, __COUNTER__)

#define INTERNAL_REGISTER_NPU_STREAM_FACTORY(factory, priority, ctr)   \
    static ::stream_executor::NpuStreamFactoryRegistrar<factory> \
    INTERNAL_REGISTER_LOCAL_NPU_STREAM_FACTORY_NAME(ctr) {             \
        priority                                                        \
    }

// __COUNTER__ must go through another macro to be properly expanded
#define INTERNAL_REGISTER_LOCAL_NPU_STREAM_FACTORY_NAME(ctr) ___##ctr##__object_

#endif  // STREAM_EXECUTOR_NPU_STREAM_FACTORY_H_
