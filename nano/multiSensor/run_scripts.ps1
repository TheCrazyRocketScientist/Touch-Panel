# THIS SCRIPT IS GENERATED USING GPT, I WILL REMOVE THIS THE FIRST CHANCE I GET.

# Run BLE scan script for 15 seconds
python ble_scanner.py -t 15

# Path to the JSON file 
$jsonFile = "addresses.json"

# Read JSON content into a PowerShell object
$devices = Get-Content $jsonFile | ConvertFrom-Json

# Iterate over each device and call multisensor.py only if the name contains 'SENSOR'
foreach ($deviceName in $devices.PSObject.Properties.Name) {
    
    if ($deviceName -like "*SENSOR*") {
        $address = $devices.$deviceName
        Write-Output "Starting multisensor.py for device '$deviceName' at address '$address'"
        Start-Process python -ArgumentList "multisensor.py -a $address -n $deviceName"
    }
    else {
        Write-Output "Skipping device '$deviceName' - does not contain 'SENSOR'"
    }
}
