from matanyone import InferenceCore


def main(
    input_path,
    mask_path,
    output_path,
    n_warmup=10,
    r_erode=10,
    r_dilate=10,
    suffix="",
    save_image=False,
    max_size=-1,
):
    processor = InferenceCore("PeiqingYang/MatAnyone")
    fgr, alpha = processor.process_video(
        input_path=input_path,
        mask_path=mask_path,
        output_path=output_path,
        n_warmup=n_warmup,
        r_erode=r_erode,
        r_dilate=r_dilate,
        suffix=suffix,
        save_image=save_image,
        max_size=max_size,
    )
    return fgr, alpha
    
