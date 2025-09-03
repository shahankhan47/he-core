
This branch is exclusively for azure prod docker deployment



build and tag  command 
docker build --no-cache -t harmonyacr-arc6c4gnfcbxfxhm.azurecr.io/hecore:170625003 .

push command 
docker push harmonyacr-arc6c4gnfcbxfxhm.azurecr.io/hecore:170625003 .

Please replace the tag for every push, the format of tag is  dd/mm/yy/xxx   where xxx is a three digit version number

The .env file must be included in the build