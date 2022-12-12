
Write-Host "Clear tmp file for python..."
Get-ChildItem -Recurse | Where-Object Name -Like '`tmp*' | Remove-Item
