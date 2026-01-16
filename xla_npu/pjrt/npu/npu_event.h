#ifndef PJRT_NPU_NPU_EVENT_H_
#define PJRT_NPU_NPU_EVENT_H_
namespace xla {

// Typical use case: AsyncValueRef<NpuEvent> is used to indicate the
// completion of a CPU operation, e.g., data transfer or running a program.
struct NpuEvent {
    NpuEvent() = default;
};

}  // namespace xla
#endif  // PJRT_NPU_NPU_EVENT_H_
