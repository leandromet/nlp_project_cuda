#!/bin/bash
# batch_process_zarr.sh - Process multiple Zarr datasets with both process_zarr.py and process_newvis.py

# Configuration options for process_zarr.py
VIS_TYPE="all"  # Visualization types to generate
MIN_PIXELS=1000 # Minimum pixels to include a class
TOP_N=15        # Number of top classes to display

# Configuration options for process_newvis.py
NEWVIS_TYPE="stacked_bar"  # stacked_bar or treemap
SECVIS_TYPE="treemap"  # Secondary visualization type
MIN_TRANSITION_PCT=1.0     # Minimum percentage for a transition to be shown
NEWVIS_TOP_N=10           # Number of top classes to display for transitions

OUTPUT_ROOT="./visualization_results"

# Create output root directory
mkdir -p "$OUTPUT_ROOT"

# Log file setup
LOG_FILE="$OUTPUT_ROOT/batch_processing.log"
echo "Batch processing started at $(date)" > "$LOG_FILE"

# Function to process a single Zarr dataset
process_zarr_dataset() {
    local zarr_path="$1"
    
    # Extract region name and grid name for organized output
    region_name=$(echo "$zarr_path" | cut -d'/' -f2)
    grid_name=$(echo "$zarr_path" | cut -d'/' -f3)
    
    # Create output directory with region and grid name
    output_dir="$OUTPUT_ROOT/${region_name}/${grid_name}"
    mkdir -p "$output_dir"
    
    echo "Processing: $zarr_path" | tee -a "$LOG_FILE"
    echo "Output to: $output_dir" | tee -a "$LOG_FILE"
    
    # Run process_zarr.py for general persistence visualizations
    echo "Running process_zarr.py..." | tee -a "$LOG_FILE"
    python ../process_zarr.py "$zarr_path" \
        --output_dir "$output_dir" \
        --vis_type "$VIS_TYPE" \
        --min_pixels "$MIN_PIXELS" \
        --top_n "$TOP_N"
    
    zarr_result=$?
    
    # Run process_newvis.py for transition visualizations
    echo "Running process_newvis.py..." | tee -a "$LOG_FILE"
    python ../process_newvis.py "$zarr_path" \
        --output_dir "$output_dir" \
        --vis_type "$NEWVIS_TYPE" \
        --min_pixels "$MIN_PIXELS" \
        --min_transition_pct "$MIN_TRANSITION_PCT" \
        --top_n "$NEWVIS_TOP_N"
    
    newvis_result=$?

    # Run process_newvis.py for transition visualizations
        echo "Running process_newvis.py..." | tee -a "$LOG_FILE"
        python ../process_newvis.py "$zarr_path" \
            --output_dir "$output_dir" \
            --vis_type "$SECVIS_TYPE" \
            --min_pixels "$MIN_PIXELS" \
            --min_transition_pct "$MIN_TRANSITION_PCT" \
            --top_n "$NEWVIS_TOP_N"
        
        newvis_result=$?

    
    
    # Check if processing was successful
    if [ $zarr_result -eq 0 ] && [ $newvis_result -eq 0 ]; then
        echo "✓ Success: $zarr_path (both visualizations)" | tee -a "$LOG_FILE"
        return 0
    elif [ $zarr_result -eq 0 ]; then
        echo "⚠ Partial success: $zarr_path (only process_zarr.py succeeded)" | tee -a "$LOG_FILE"
        return 1
    elif [ $newvis_result -eq 0 ]; then
        echo "⚠ Partial success: $zarr_path (only process_newvis.py succeeded)" | tee -a "$LOG_FILE"
        return 1
    else
        echo "✗ Failed: $zarr_path (both visualizations failed)" | tee -a "$LOG_FILE"
        return 2
    fi
}

# List of all Zarr datasets to process
datasets=(
    "./PIQUIRI_PARANÁ_2/grid_54.112151W_51.600066W_25.310709S_23.652716S/data.zarr"
    "./PIQUIRI_PARANÁ_2/grid_54.093373W_53.372756W_24.024587S_23.307052S/data.zarr"
    "./LITORÂNEA/grid_49.130488W_48.082280W_25.993693S_24.977702S/data.zarr"
    "./BAIXO_TIBAGI/grid_51.458814W_50.477205W_24.103366S_22.799724S/data.zarr"
    "./PARANÁ_3/grid_54.619173W_53.440380W_25.591544S_24.028442S/data.zarr"
    "./PIRAPÓ_PARANAPANEMA_3_PARANAPANEMA_4/grid_52.213447W_51.383543W_23.605040S_22.542408S/data.zarr"
    "./PIRAPÓ_PARANAPANEMA_3_PARANAPANEMA_4/grid_53.089144W_52.029525W_23.178869S_22.519304S/data.zarr"
    "./PIRAPÓ_PARANAPANEMA_3_PARANAPANEMA_4/grid_52.027487W_51.017027W_23.320778S_22.542571S/data.zarr"
    "./AFLUENTES_DO_MÉDIO_IGUAÇU/grid_52.386764W_50.396319W_26.715857S_25.156711S/data.zarr"
    "./Corpos_d'Água/grid_54.619507W_48.023527W_25.892099S_22.516556S/data.zarr"
    "./ALTO_TIBAGI/grid_51.052736W_49.635514W_25.656299S_23.991435S/data.zarr"
    "./ITARARÉ_CINZAS_PARANAPANEMA_1_PARANAPANEMA_2/grid_49.874916W_49.200879W_24.646770S_23.154931S/data.zarr"
    "./ITARARÉ_CINZAS_PARANAPANEMA_1_PARANAPANEMA_2/grid_50.527186W_49.709772W_23.286508S_22.900600S/data.zarr"
    "./ITARARÉ_CINZAS_PARANAPANEMA_1_PARANAPANEMA_2/grid_50.997343W_50.528735W_23.179854S_22.795784S/data.zarr"
    "./ITARARÉ_CINZAS_PARANAPANEMA_1_PARANAPANEMA_2/grid_50.650983W_49.783707W_24.465312S_22.936455S/data.zarr"
    "./ALTO_IGUAÇU_AFLUENTES_DO_RIO_NEGRO_AFLUENTES_DO_RIO_RIBEIRA/grid_49.985577W_48.208767W_25.476827S_24.401789S/data.zarr"
    "./ALTO_IGUAÇU_AFLUENTES_DO_RIO_NEGRO_AFLUENTES_DO_RIO_RIBEIRA/grid_50.496043W_48.954241W_26.237773S_25.222908S/data.zarr"
    "./BAIXO_IVAÍ_PARANÁ_1/grid_53.696929W_51.970663W_24.069602S_22.906527S/data.zarr"
    "./BAIXO_IVAÍ_PARANÁ_1/grid_53.708471W_53.033501W_23.302636S_22.661396S/data.zarr"
    "./ALTO_IVAÍ/grid_52.523467W_50.743260W_25.582211S_23.414066S/data.zarr"
    "./AFLUENTES_DO_BAIXO_IGUAÇU/grid_54.591987W_51.509911W_26.601824S_24.942000S/data.zarr"
)

# Process each dataset
echo "Starting batch processing of ${#datasets[@]} Zarr datasets..." | tee -a "$LOG_FILE"
echo "-------------------------------------------" | tee -a "$LOG_FILE"

total_datasets=${#datasets[@]}
fully_successful=0
partially_successful=0
failed=0

for dataset in "${datasets[@]}"; do
    # Process the dataset
    process_zarr_dataset "$dataset"
    result=$?
    
    # Update counters based on success/failure
    if [ $result -eq 0 ]; then
        ((fully_successful++))
    elif [ $result -eq 1 ]; then
        ((partially_successful++))
    else
        ((failed++))
    fi
    
    echo "-------------------------------------------" | tee -a "$LOG_FILE"
done

# Print summary
echo "Batch processing completed at $(date)" | tee -a "$LOG_FILE"
echo "Total datasets: $total_datasets" | tee -a "$LOG_FILE"
echo "Fully successful: $fully_successful" | tee -a "$LOG_FILE"
echo "Partially successful: $partially_successful" | tee -a "$LOG_FILE"
echo "Failed: $failed" | tee -a "$LOG_FILE"
echo "All results saved to: $OUTPUT_ROOT" | tee -a "$LOG_FILE"

# Make the script executable with: chmod +x batch_process_zarr.sh