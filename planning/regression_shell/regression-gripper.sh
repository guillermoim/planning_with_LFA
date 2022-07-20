domain="gripper"

echo "First instance"
instance="prob01_0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50

echo "Second instance"
instance="prob01_1-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50

echo "Third instance"
instance="prob02_0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50

echo "Fourth instance"
instance="prob02_1-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50

