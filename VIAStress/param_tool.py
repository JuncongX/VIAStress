def exp_param_tool(args):
    if args.dataset_name == 'wesad':
        if args.y_dim == 2:
            return 1
        else:
            return 0.5
    elif args.dataset_name == 'ubfc_phys':
        if args.ubfc_phys_task == 2:
            return 1
        else:
            return 0.5
    elif args.dataset_name == 'can_stress':
        return 1
    elif args.dataset_name == 'verbio':
        return 1.5


def out_dim_tool(args):
    return 64
    # if args.dataset_name == 'wesad':
    #     if args.y_dim == 2:
    #         return 128
    #     else:
    #         return 64
    # elif args.dataset_name == 'ubfc_phys':
    #     if args.ubfc_phys_task == 2:
    #         return 128
    #     else:
    #         return 128


def distance_p(args):
    if args.dataset_name == 'wesad':
        if args.y_dim == 2:
            return 2
        else:
            return 2
    elif args.dataset_name == 'ubfc_phys':
        if args.ubfc_phys_task == 2:
            return 1
        else:
            return 2
    elif args.dataset_name == 'can_stress':
        return 1
    elif args.dataset_name == 'verbio':
        return 1
