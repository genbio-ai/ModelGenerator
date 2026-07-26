import os
import sys

def reassemble_chunks(output_file, chunks_dir):
    with open(output_file, "wb") as f_out:
        for chunk_file in sorted(os.listdir(chunks_dir)):
            chunk_path = os.path.join(chunks_dir, chunk_file)
            with open(chunk_path, "rb") as f_in:
                f_out.write(f_in.read())
    print(f"Reassembled model saved to {output_file}")


if len(sys.argv) < 3:
    print("Usage: python merge_ckpt.py <model_chunk_dir>  <merged_model_path>")
    sys.exit(1)

# Parameters
chunks_dir = sys.argv[1] #"model_chunks"  # Directory containing downloaded chunks
output_file = sys.argv[2]

reassemble_chunks(output_file, chunks_dir)
