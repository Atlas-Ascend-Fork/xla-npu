//
// Created by m00526841 on 25-12-19.
//

#ifndef NPU_EVENT_H_
#define NPU_EVENT_H_
namespace xla {

// Typical use case: AsyncValueRef<NpuEvent> is used to indicate the
// completion of a CPU operation, e.g., data transfer or running a program.
struct NpuEvent {
  NpuEvent() = default;
};

}  // namespace xla
#endif //NPU_EVENT_H_
