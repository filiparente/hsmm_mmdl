function state = sampleDiscrete(p)
    landings = mnrnd(1, p, 1);
    state = find(landings == 1);
end