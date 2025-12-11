from molguidance.utils.sdf_utils import get_pb_valid_results 

pb_vailla, _ = get_pb_valid_results("../vanilla/vanilla.sdf")
pb_cfg,_ = get_pb_valid_results("../cfg/cfg_guidance_best_w.sdf")
pb_ag, _ = get_pb_valid_results("../ag/ag_guidance_best_w.sdf")
pb_mg, _ = get_pb_valid_results("../mg/mg_guidance_best_w.sdf")

print("PoseBuster valid results:")
print(f"pb_vailla: {len(pb_vailla)/10000}")
print(f"pb_cfg: {len(pb_cfg)/10000}")
print(f"pb_ag: {len(pb_ag)/10000}")
print(f"pb_mg: {len(pb_mg)/10000}")