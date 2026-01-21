#include "xla/xla_npu/stream_executor/npu_stream_factory.h"

#include <memory>
#include <utility>

#include "absl/base/const_init.h"
#include "absl/synchronization/mutex.h"

namespace {

static absl::Mutex* get_npu_stream_factory_lock() {
    static absl::Mutex npu_stream_factory_lock(absl::kConstInit);
    return &npu_stream_factory_lock;
}

struct FactoryItem {
    std::unique_ptr<stream_executor::NpuStreamFactory> factory;
    int priority = -1;
};

FactoryItem& npu_stream_factory() {
    static FactoryItem* const factory = new FactoryItem();
    return *factory;
}

}  // namespace

namespace stream_executor {

// static
void NpuStreamFactory::Register(std::unique_ptr<NpuStreamFactory> factory, int priority) {
    absl::MutexLock l(*get_npu_stream_factory_lock());
    FactoryItem& factory_item = npu_stream_factory();
    if (factory_item.factory == nullptr) {
        factory_item.factory = std::move(factory);
        factory_item.priority = priority;
        return;
    }
    if (factory_item.priority < priority) {
        factory_item.factory = std::move(factory);
        factory_item.priority = priority;
    }
}

// static
const NpuStreamFactory* NpuStreamFactory::GetFactory() {
    absl::ReaderMutexLock l(*get_npu_stream_factory_lock());
    FactoryItem& factory_item = npu_stream_factory();
    return factory_item.factory.get();
}

}  // namespace stream_executor

REGISTER_NPU_STREAM_FACTORY(stream_executor::NpuStreamDefaultFactory, 100);
