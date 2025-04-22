import logging
import zarr

def test_zarr_integrity(zarr_path):
    try:
        root = zarr.open(zarr_path)
        logging.info(f"Zarr shape: {root['data'].shape}")
        logging.info(f"First pixel: {root['data'][0,0,0]}")
        return True
    except:
        logging.error("Zarr file corrupted!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    zarr_path = "/media/bndt-ai/db_lin/gitstore/leandromet/nlp_project_cuda/raster_proc/combined_results/combined_data.zarr"
    test_zarr_integrity(zarr_path)
    logging.info("Zarr integrity test completed.")

