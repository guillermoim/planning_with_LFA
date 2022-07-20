domain="visitall"

echo "First instance"
instance="p-1-5_0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 10

