**目前程式流程 更新至3/27**
[notion link](https://www.notion.so/anfangli/TouchSDF-327061b380d080e3bf95f0403c8bcd9f)
1. 收資料
    
    ```bash
    conda activate TouchSDF
    cd Desktop/TouchSDF
    
    python scripts/stage1_tactile_collect_mesh_filter.py
    # 會濾掉重複區域的版本
    
    python scripts/stage1_tactile_collect_mesh.py
    ```
    
2. 所有資料進行座標系對齊(逐一)
    
    ```bash
    conda activate deepsdf
    cd Desktop/TouchSDF
    
    python scripts/reconstruct_from_points_sdf_sequence.py   --test_dir /home/anfang/Desktop/TouchSDF/results/runs_touch_sdf/26_03_162029_5467/   --points_name points_sdf.pt   --mode cumulative   --warm_start   --save_observation_npy
    ```
    
3. 資料生成(以原資料為基準作為面，再衍生出更多點)
    
    ```bash
    conda activate deepsdf
    cd Desktop/TouchSDF
    
    python scripts/prepare_points_sdf_per_touch_chart.py   --test_dir /home/anfang/Desktop/TouchSDF/results/runs_touch_sdf/26_03_162029_5467/   --output_name points_sdf.pt   --save_merge
    ```
    
4. 迴圈形式逐一輸入資料，一步一步完善3D圖檔
    
    ```bash
    conda activate deepsdf
    cd ../DeepSDF/
    
    python scripts/reconstruct_from_points_sdf_sequence.py   --test_dir /home/anfang/Desktop/TouchSDF/results/runs_touch_sdf/26_03_162029_5467/   --points_name points_sdf.pt   --mode cumulative   --warm_start   --save_observation_npy
    ```
    
- 看看效果
    1. `/home/anfang/Desktop/TouchSDF/results/runs_touch_sdf/{dataset編號}/infer_from_points_sdf_cumulative/{data_index}/output_mesh.obj` 找可查閱軟體查看
    ex. https://3dviewer.net/index.html
    2. 對齊看效果
        
        ```bash
        python visualize_alignment_flexible.py \
           --input_a {a圖位址} \
           --label_a {a圖標籤} \
        	 --input_b {b圖位址} \
           --label_b {b圖標籤} \
           --input_c {c圖位址} \
           --label_c {c圖標籤} \
           --out_dir {照片輸出位址}
        # 可以更多圖一起比較，按照格式寫就好
        
        # 我的用法
        python visualize_alignment_flexible.py
           --input_a /home/anfang/Desktop/TouchSDF/results/runs_touch_sdf/19_03_102750_1924/mesh_deepsdf.obj \
           --label_a obj_mesh \
           --input_b /home/anfang/Desktop/TouchSDF/results/runs_touch_sdf/19_03_102750_1924/completion_data_stage1_gt.pt \
           --label_b touch_complete_ply \
           --input_c /home/anfang/Desktop/DeepSDF/results/runs_sdf/17_07_172540/infer_latent_19_03_102914/output_mesh.obj \
           --label_c deepsdf_mesh \
           --out_dir /home/anfang/Downloads/
        ```
        
