# Báo cáo bài nộp — Day 23 Sensor Fusion Lab

> Điền file này rồi commit. Cách nộp: [hướng dẫn nộp](../SUBMISSION.md).

## Thông tin học viên

- Họ tên: Pham Van Hoang Anh Tu
- MSSV: 2A202602507
- Email:
- Link repo (fork): https://github.com/tubepvhat1604-bot/K4-L2L3-DAY23-PhamVanHoangAnhTu-2A202602507-SensorFusion
- Commit hash nộp (`git rev-parse HEAD`):

## Tóm tắt kết quả

- `fusion_mode` (bắt buộc `compare`), `frames`, `segment`, `seed`:
- `detection.precision`, `detection.recall`, `detection.tp/fp/fn`:
- `tracking.lidar.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
- `tracking.fused.rmse`, `matches`, `sum_sq_err`, `ghost_track_frames`, `missed_gt_frames`, `mean_confirmed_tracks`:
- Giải thích khác biệt hai mode, đọc RMSE cùng số ghép và ghost/miss:

Chạy từ root repo:

```bash
fusion-run-lab --config student/config/paths.yaml --fusion compare --seed 0
```

`rmse = sqrt(sum_sq_err/matches)` trên vị trí 3D của confirmed tracks ghép
một-một với GT xe trong cửa sổ BEV, gate XY **2.0 m**; `null` nếu không có cặp.
Camera dùng tâm hộp 2D ground-truth FRONT có nhiễu seeded, **không** dùng camera
detector. Kết quả này không đo hiệu quả một perception system độc lập với GT.

`grade_run.log` là JSONL, mỗi `(mode,frame)` đúng một record với các trường:
`mode`, `frame`, `det_tp`, `det_fp`, `det_fn`, `valid_gt`, `confirmed`, `matches`,
`sum_sq_err`, `ghosts`, `misses`. Đảm bảo `matches+ghosts==confirmed` và
`matches+misses==valid_gt`; tổng/trung bình record phải khớp `metrics.json`.
File per-mode `metrics_lidar.json`, `metrics_fused.json`, `grade_run_lidar.log`,
`grade_run_fused.log` được giữ để đối chiếu.

## Giải thích ngắn (Parts E–H — tự viết)

1. Khác biệt đo lidar 3D và camera 2D trong EKF (`z`, `R`)?

   LiDAR đo trực tiếp vị trí tâm hộp: `z = (x, y, z)` (mét, 3×1), `R = diag(0.1², 0.1², 0.1²)`.
   `h(x)` là phép biến đổi tuyến tính sang hệ cảm biến nên `H` là ma trận hằng 3×6
   (khối xoay 3×3, cột vận tốc bằng 0). Camera đo toạ độ pixel `z = (u, v)` (2×1),
   `R = diag(5², 5²)` px² (`build_camera_measurement`). `h(x)` là phép chiếu pinhole
   `u = c_i − f_i·y_s/x_s`, `v = c_j − f_j·z_s/x_s` (`camera_measurement_prediction`), phi
   tuyến theo độ sâu `x_s`, nên EKF dùng Jacobian `H` 2×6 tính lại tại trạng thái hiện tại.
   Camera không đo độ sâu nên chỉ hiệu chỉnh tốt hướng ngang/dọc, LiDAR hiệu chỉnh cả 3 trục.

2. Vì sao cần gating Mahalanobis trước khi gán?

   `d² = γᵀ S⁻¹ γ` đo khoảng cách theo độ bất định `S = H P Hᵀ + R`, nên một sai lệch 1 m
   là "gần" với track mới sinh (P lớn) nhưng "xa" với track đã hội tụ. Cổng χ²
   (`chi2.ppf(0.995, dim_meas)`, ~12.8 cho LiDAR 3D, ~10.6 cho camera 2D, xem `chi2_gate`)
   loại các cặp gần như chắc chắn không cùng vật trước khi greedy chọn cặp rẻ nhất.
   Không có cổng thì một detection nhiễu hoặc xe bên cạnh vẫn có thể bị gán vào track,
   kéo trạng thái lệch đi và sinh đổi ID; detection bị loại khỏi cổng thì thành track mới.

3. Pipeline là track-then-fuse hay fuse-then-track? Chỉ ra trên log `fusion-run-lab`.

   Track-then-fuse. Trong `platform/fusion_lab/scripts/run_lab.py`, mỗi frame: `KF.predict`
   một lần cho mọi track, rồi `assoc.associate_and_update(..., lidar_sensor)`, rồi nếu là mode
   fused thì `assoc.associate_and_update(..., camera_sensor)` trên cùng `manager`. Chỉ có một
   tracker; camera không tạo detection riêng để ghép trước khi track. Trên `grade_run.log`,
   các trường `det_tp/det_fp/det_fn` giống hệt nhau giữa record `"mode": "lidar"` và
   `"mode": "fused"` cùng frame (detection chỉ từ LiDAR), chỉ các trường tracking
   (`confirmed`, `matches`, `sum_sq_err`, `ghosts`, `misses`) khác nhau vì camera chỉ
   tinh chỉnh trạng thái track đã có.

4. Nếu camera lệch calibration, triệu chứng gì trên innovation/residual?

   Innovation camera `γ = z − h(x)` sẽ có độ lệch có hệ thống (bias) cùng dấu qua nhiều frame
   và nhiều track thay vì dao động quanh 0, và `d²` trung bình tăng cao hơn kỳ vọng
   (≈ dim_meas = 2). Lệch nhỏ: cặp vẫn qua cổng χ² và update kéo vị trí track về phía sai,
   RMSE fused tăng so với LiDAR, LiDAR lại kéo ngược về ở frame sau. Lệch lớn: `d²` vượt
   cổng, cặp camera bị loại và camera gần như không còn tác dụng (fused ≈ lidar).

5. Vì sao `associate_and_update(..., sensor)` cần sensor tường minh ở frame rỗng?
   Giải thích vì sao lidar quyết định score/init/delete còn camera chỉ EKF update.

   Khi `meas_list` rỗng thì không thể suy ra sensor từ `meas.sensor`, mà `manage_tracks`
   vẫn phải biết đây là lượt LiDAR để trừ score các track trong FOV bị bỏ lỡ và xoá track
   hết score; nếu không, một frame LiDAR không có detection sẽ bị bỏ qua và ghost track
   sống mãi. Ngược lại, nếu nhầm lượt camera là LiDAR thì track bị trừ điểm sai.
   LiDAR là cảm biến duy nhất có detector thật cho hộp 3D, phủ 360° và có độ sâu, nên nó
   quyết định tồn tại (`handle_updated_track` và `manage_tracks` chỉ chạy nhánh lidar).
   Camera ở đây là tâm hộp GT có nhiễu, chỉ thấy phía trước, không có độ sâu nên không
   khởi tạo được trạng thái 3D; cho camera cộng/trừ score sẽ làm xe ngoài FOV camera bị
   xoá oan hoặc xe chỉ có trong camera được xác nhận mà không có vị trí đúng.

6. Nêu điều kiện xác nhận, giữ confirmed sau miss, và điều kiện xóa track.

   Track mới: `score = 1/window = 1/6`, state `initialized` (`init_track_state_from_meas`).
   Mỗi lượt LiDAR: hit thì `score += 1/6` (tối đa 1) và state thành `tentative`; miss trong
   FOV LiDAR thì `score −= 1/6`. Xác nhận khi `score > confirmed_threshold = 0.8`, tức sau
   4 hit LiDAR liên tiếp kể từ frame khởi tạo (score 5/6 ≈ 0.83). Track đã `confirmed` không bị hạ trạng thái khi miss
   (`update_track_score` return sớm). Xoá (`should_delete_track`, điều kiện OR): `P[0,0]` hoặc
   `P[1,1] > max_P = 9` (độ bất định vị trí > 3 m), hoặc track confirmed có
   `score < delete_threshold = 0.6`, hoặc track chưa confirmed có `score ≤ 0`.

## Bonus (không bắt buộc)

Liệt kê phần bonus đã làm, file bằng chứng trong `student/bonus/` và kết quả chính
(xem [RUBRIC.md](../RUBRIC.md) mục 2). Không làm thì ghi "Không".

- Không

## Khai báo sử dụng AI (bắt buộc)

Ghi rõ, kể cả khi không dùng ("Không dùng AI"). Xem [RULES.md](../RULES.md) mục 2.

- Công cụ đã dùng (ChatGPT, Copilot, Claude, …): Claude (Claude Code, Anthropic)
- Dùng cho phần nào (hàm, câu hỏi, debug): Claude viết code Part E–H trong `student/workspace/` (`kalman.py`, `camera_fusion.py`, `association.py`, `track_management.py`) và soạn nháp 6 câu giải thích E–H; tôi đã đọc lại và chỉnh theo hiểu biết của mình. Số liệu trong phần "Tóm tắt kết quả" lấy từ `metrics.json` của lần chạy của tôi.
- Cách bạn đã kiểm tra lại (pytest, chạy Waymo, đối chiếu công thức): `pytest student/tests -q` (128 passed, không failed/xfailed); đối chiếu công thức với `docs/HUONG_DAN_KY_THUAT.md` và các gợi ý `# vi: TODO`; chạy `fusion-run-lab --fusion compare --seed 0` trên frame 0–198 và `python tools/check_submission.py`.

## Checklist nộp

- [ ] **Part E–H** trong `workspace/` đã implement; `pytest student/tests -q` không còn `failed`/`xfailed`
- [ ] Part A–D: không bắt buộc sửa (hoặc ghi chú nếu bạn đã sửa)
- [ ] Lần chạy chấm điểm: `--fusion compare --seed 0`, `frame_start: 0`, `frame_end: 198`
- [ ] Đã commit `student/artifacts/metrics*.json` và `student/artifacts/grade_run*.log` (không sửa tay)
- [ ] Đã điền đủ file này, gồm khai báo AI
- [ ] Không commit dữ liệu Waymo, weights, `paths.yaml`, API key
- [ ] `python tools/check_submission.py` báo `KẾT QUẢ: SẴN SÀNG NỘP`
- [ ] Đã push và nộp link repo + commit hash trên LMS ([hướng dẫn nộp](../SUBMISSION.md))
