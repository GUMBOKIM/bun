/// Runs `run` compiled for BMI2 if the CPU has it. That holds for what is inlined: `run` and its callees are
/// `#[inline(always)]`.
#[cfg(all(target_arch = "x86_64", not(miri)))]
#[inline(always)]
#[expect(unsafe_code)]
pub fn with_bmi2<R>(run: impl FnOnce() -> R) -> R {
    #[target_feature(enable = "bmi1,bmi2,lzcnt")]
    fn bmi2<R>(run: impl FnOnce() -> R) -> R {
        run()
    }

    if is_x86_feature_detected!("bmi1")
        && is_x86_feature_detected!("bmi2")
        && is_x86_feature_detected!("lzcnt")
    {
        // SAFETY: the CPU has the three features.
        unsafe { bmi2(run) }
    } else {
        run()
    }
}

#[cfg(not(all(target_arch = "x86_64", not(miri))))]
#[inline(always)]
pub fn with_bmi2<R>(run: impl FnOnce() -> R) -> R {
    run()
}
