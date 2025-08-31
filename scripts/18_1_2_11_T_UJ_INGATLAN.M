
   

    #"Filtered Row For Új lakásoka" = Table.SelectRows(#"Added Index", each Text.Contains([Régió], "Új lakásoka")),

    #"Row Index for Új lakásoka" = if Table.IsEmpty(#"Filtered Row For Új lakásoka") 
        then null 
        else List.Min(#"Filtered Row For Új lakásoka"[Index]),

    #"Skipped Rows" = if #"Row Index for Új lakásoka" = null 
        then #table({}, {}) 
        else Table.Skip(#"Added Index", #"Row Index for Új lakásoka" + 1),



