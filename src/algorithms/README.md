# TODO A*
A* is probably better to figure out where to micro units to using influence maps

- Add usage of path cost in grid
    - 0 = wall
    - 1 = standard path
    - \>1 path with increased cost
    
- Add more function parameters
    - Abort path-find once closed_set reaches a number of points
    - Abort path-find once distance to start reached a certain distance
    - Construct path once distance to goal reached `distance < 5`, e.g. units with ranged attack just need to get in range, and not to the target location which probably results in shorter path
    
- Add helper functions
    - pf.calc_path_goals(source: Point, goals: List[Point]) -> Tuple[Point, List[Point]]
        - calculate which goal is closest (by path) to source, and return path
    
- Add more tests
    - Calculate path and check if it matches exact expected path
    - Find path over cliff (increased cost on the mountain, decreased cost down the slope?)
    

# TODO JPS
JPS is probably more ideal for long travel distances or to find out if the enemy walled off the entrance entirely

- Add more general functions
    - pf.path_exists(source, goal) -> bool
        - fast path checking without creating the path vector / list
    - pf.calc_path(source, goal) -> List[Point]
        - calculate one path
    - pf.calc_paths(sources: List[Point], goals: List[Point]) -> List[List[Point]]
        - calculate multiple paths
    - a function that takes a numpy array and converts it to ndarray to be used in JPS

- Add more tests
    - Export more map data to test
    
    
    
    