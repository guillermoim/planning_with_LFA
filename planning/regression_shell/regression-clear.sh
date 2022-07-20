domain="blocksworld"

echo "3 blocks"
instance="p-clear-3blocks-0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50
exit;
echo "4 blocks"
instance="p-clear-4blocks-0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50

echo "5 blocks"
instance="p-clear-5blocks-0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50

echo "6 blocks"
instance="p-clear-6blocks-0-flag"
python regression.py --filepath results/$domain/$instance/feat_matrix_extended.csv --step 50
