python3 -m venv venv
source venv/bin/activate

pip install -r requirements.txt

python -m paralysis_recovery_test.run my_new_banner learn
python -m paralysis_recovery_test.run my_new_banner tune
python -m paralysis_recovery_test.run my_new_banner test
python -m paralysis_recovery_test.run my_new_banner eval
