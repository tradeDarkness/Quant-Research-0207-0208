
import qlib
from qlib.data import D
from qlib.config import REG_US
import os

provider_uri = os.path.expanduser("~/.qlib/qlib_data/my_crypto")
qlib.init(provider_uri=provider_uri, region=REG_US)

def test_loader():
    instruments = ["BTCUSDT"]
    fields = ["$close", "$volume", "$open", "$high", "$low"]
    start_time = "2024-03-01 00:00:00"
    end_time = "2024-03-02 00:00:00"
    freq = "15min" 

    print(f"Loading features for {instruments} from {start_time} to {end_time} freq={freq}...")
    
    try:
        # Check Instrument Listing
        from qlib.data import D
        inst_dict = D.list_instruments(instruments=D.instruments(market="all"), freq=freq, as_list=False)
        print(f"All Instruments Dict at {freq}: {inst_dict}")
        if 'BTCUSDT' in inst_dict:
            print(f"BTCUSDT Range: {inst_dict['BTCUSDT']}")
        
        # Check calendar
        cal = D.calendar(freq=freq, future=False)
        print(f"Calendar[{freq}] head: {cal[:5]}")
        
        df = D.features(instruments, fields, start_time, end_time, freq=freq)
        print("Data Loaded Successfully!")
        print(df)
    except Exception as e:
        print(f"Error loading data: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_loader()
