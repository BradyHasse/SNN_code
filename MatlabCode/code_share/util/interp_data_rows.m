function newData = interp_data_rows(data, times, newTimes)
    [numRow, numCol] = size(data);
    
    newData = nan(numRow, length(newTimes));
    for irow = 1:numRow
        y = data(irow, :);
        newy = interp1(times, y, newTimes, 'spline', 'extrap');
        newData(irow, :) = newy;
    end
end