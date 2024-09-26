function par = get_params(parfile)
par = jsondecode(textread(parfile,'%c'));
end
