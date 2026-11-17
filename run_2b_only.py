import sys
sys.path.append(r'C:\Users\sayan\OneDrive\Desktop\projects\d1-browser-agent')
import compare_600m_vs_2b as b

m2b, load2b, vram2b = b.load_model(b.MODEL_2B, quant_4bit=True)
b.print(f'\n>>> MODEL B: d1-2B (4-bit NF4) | Load: {load2b:.2f}s | Peak VRAM: {vram2b:.1f} MB')
b.run_browser_eval(m2b, 'd1-2B (4-bit NF4)')
b.run_query_extraction_eval(m2b, 'd1-2B (4-bit NF4)', supports_read_text=True)
b.run_joint_single_pass_test(m2b, 'd1-2B (4-bit NF4)', supports_read_text=True)
b.run_safety_and_calibration(m2b, 'd1-2B (4-bit NF4)')
b.run_multimodal_ui_test(m2b, 'd1-2B (4-bit NF4)', is_2b=True)
b.run_batch_throughput(m2b, 'd1-2B (4-bit NF4)')
b.print('=== ALL 2B BENCHMARKS COMPLETE ===')
