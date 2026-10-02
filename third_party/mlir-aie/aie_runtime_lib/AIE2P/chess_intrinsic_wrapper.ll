; ModuleID = '/__w/mlir-aie/mlir-aie/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.cpp'
source_filename = "/__w/mlir-aie/mlir-aie/aie_runtime_lib/AIE2P/chess_intrinsic_wrapper.cpp"



%struct.ipd.custom_type.uint2_t.uint2_t = type { i2 }

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2p___acquire(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_acquire____uint___uint(i32 zeroext %0, i32 zeroext %1) #3
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: mustprogress nounwind
define dso_local void @llvm___aie2p___release(i32 noundef %0, i32 noundef %1) local_unnamed_addr addrspace(1) #0 {
  tail call addrspace(1) void @llvm.chess_memory_fence()
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_release____uint___sint(i32 zeroext %0, i32 signext %1) #3
  tail call addrspace(1) void @_Z25chess_separator_schedulerv() #3
  tail call addrspace(1) void @llvm.chess_memory_fence()
  ret void
}

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
define dso_local void @llvm___aie___event0() local_unnamed_addr addrspace(1) #1 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t zeroinitializer) #3
  ret void
}

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
define dso_local void @llvm___aie___event1() local_unnamed_addr addrspace(1) #1 {
  tail call x86_regcallcc addrspace(1) void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t { i2 1 }) #3
  ret void
}

; Function Attrs: mustprogress nounwind willreturn
declare void @llvm.chess_memory_fence() addrspace(1) #2

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local void @_Z25chess_separator_schedulerv() local_unnamed_addr addrspace(1) #1

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_acquire____uint___uint(i32 zeroext, i32 zeroext) local_unnamed_addr addrspace(1) #1

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_release____uint___sint(i32 zeroext, i32 signext) local_unnamed_addr addrspace(1) #1

; Function Attrs: nounwind memory(inaccessiblemem: readwrite)
declare dso_local x86_regcallcc void @__regcall3__chessintr_void_event_uint2_t(%struct.ipd.custom_type.uint2_t.uint2_t) local_unnamed_addr addrspace(1) #1

attributes #0 = { mustprogress nounwind "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #1 = { nounwind memory(inaccessiblemem: readwrite) "frame-pointer"="all" "no-builtin-memcpy" "no-trapping-math"="true" "stack-protector-buffer-size"="8" }
attributes #2 = { mustprogress nounwind willreturn }
attributes #3 = { nounwind memory(inaccessiblemem: readwrite) "no-builtin-memcpy" }

!llvm.linker.options = !{}
!llvm.chess.memory-units = !{!0, !1, !2, !3, !4, !5, !6, !7, !8, !9, !10, !11, !12, !13}
!llvm.module.flags = !{!14, !15}
!llvm.ident = !{!16}

!0 = !{i32 0, i8 undef}
!1 = !{i32 2, i8 undef}
!2 = !{i32 3, i8 undef}
!3 = !{i32 4, i8 undef}
!4 = !{i32 5, i8 undef}
!5 = !{i32 6, i8 undef}
!6 = !{i32 7, i8 undef}
!7 = !{i32 8, i8 undef}
!8 = !{i32 9, i8 undef}
!9 = !{i32 10, i8 undef}
!10 = !{i32 11, i8 undef}
!11 = !{i32 12, i8 undef}
!12 = !{i32 13, i8 undef}
!13 = !{i32 14, i8 undef}
!14 = !{i32 1, !"wchar_size", i32 4}
!15 = !{i32 7, !"frame-pointer", i32 2}
!16 = !{!"clang version 16.0.3 (/u/sgasip/ipd/repositories/llvm_ipd 3bf57c65bea6bc9606f00be3c91a3465230e34ae)"}
