#THIS SCRIPT IS GENERATED USING GPT, I WILL REMOVED THIS THE FIRST CHANCE I GET.

# Run BLE scan script for 5 seconds
python ble_scanner.py -t 5

# Path to the JSON file 
$jsonFile = "addresses.json"

# Read JSON content into a PowerShell object
$devices = Get-Content $jsonFile | ConvertFrom-Json

# Iterate over each device and call multisensor.py
foreach ($deviceName in $devices.PSObject.Properties.Name) {

    $address = $devices.$deviceName
    Write-Output "Starting multisensor.py for device '$deviceName' at address '$address'"
    
    Start-Process python -ArgumentList "multisensor.py -a $address -n $deviceName"
}
