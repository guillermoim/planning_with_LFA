domain="transport"

echo "First instance"
instance="p01-opt08_0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50

