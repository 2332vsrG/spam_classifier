# auto_download_dataset.py - FASTEST SOLUTION
# Save this in your spam_classifier folder and run it!

import os
import urllib.request
import pandas as pd

print("=" * 70)
print("AUTO DATASET DOWNLOADER - Email Spam Classifier")
print("=" * 70)

# Create datasets folder
dataset_path = "spam_dataset.csv"

try:
    print("\n📥 Downloading SMS Spam Collection Dataset...")
    print("This may take 30-60 seconds...\n")
    
    # Download from UCI Repository
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = "smsspamcollection.zip"
    
    # Download zip file
    urllib.request.urlretrieve(url, zip_path)
    print("✓ Downloaded zip file")
    
    # Extract zip
    import zipfile
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall()
    print("✓ Extracted files")
    
    # Read SMSSpamCollection (no header)
    df = pd.read_csv(
        "SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8"
    )
    
    print(f"✓ Loaded {len(df)} messages")
    
    # Save as spam_dataset.csv
    df.to_csv(dataset_path, index=False)
    print(f"✓ Saved to: {dataset_path}")
    
    # Verify
    verify_df = pd.read_csv(dataset_path)
    print(f"\n✅ DATASET READY!")
    print(f"   Total messages: {len(verify_df)}")
    print(f"   Ham messages: {len(verify_df[verify_df['label'] == 'ham'])}")
    print(f"   Spam messages: {len(verify_df[verify_df['label'] == 'spam'])}")
    print(f"   Location: {os.path.abspath(dataset_path)}")
    
    # Cleanup
    os.remove(zip_path)
    os.remove("SMSSpamCollection")
    print("\n✓ Cleaned up temporary files")
    
    print("\n" + "=" * 70)
    print("NEXT STEPS:")
    print("=" * 70)
    print("1. Run: python model_trainer.py")
    print("2. Wait for training to complete")
    print("3. You should see Accuracy: 0.98+ (98%+)")
    print("=" * 70 + "\n")
    
except Exception as e:
    print(f"\n❌ ERROR: {e}")
    print("\nTROUBLESHOOTING:")
    print("1. Check internet connection")
    print("2. Make sure you're in the correct folder")
    print("3. Try running again")
    print("\nIF THIS FAILS, use alternative method below...")
    
print("\n" + "=" * 70)
print("ALTERNATIVE: Manual Dataset Upload")
print("=" * 70)
print("""
If auto-download doesn't work:

1. Go to: https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset
2. Click 'Download' button (top right)
3. Extract the downloaded spam.csv
4. Move spam.csv to your spam_classifier folder
5. Rename to: spam_dataset.csv
6. Run: python model_trainer.py
""")
print("=" * 70)