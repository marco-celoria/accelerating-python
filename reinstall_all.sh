mkdir -p logs

job1=$(sbatch --parsable submit_install_1.sh)
job1=${job1%%;*}
echo "Submitted stage 1 as job ${job1}"

job2=$(
    sbatch \
        --parsable \
        --dependency="afterok:${job1}" \
        submit_install_2.sh
)
job2=${job2%%;*}
echo "Submitted stage 2 as job ${job2}"
