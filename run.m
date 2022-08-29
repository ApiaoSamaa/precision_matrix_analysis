p = 100;

% p = 20, k = 43: 399
% p = 40, k = 186: 1599
% p = 60, k = 421: 3599
% p = 100, k = 1086: 9999
% p = 200, k = 5108: 40000
% p = 300, k = 10284: 90000
for k = 9884: 100: 9984
    %jsonData = loadjson("./cancer/output_dimension"+string(p)+"_"+string(k)+"_cov.json");
    jsonData = loadjson("../mat_cov/fastmdmc_input/dimension"+string(p)+"nonzero"+string(k)+"fastmdmc_input_cov.json");
    cov = zeros(p);
    for i = 1: p
        for j = 1: p
            cov(i, j) = jsonData(i, j);
        end
    end

    fastmdmc(sparse(cov), k);
               
end


