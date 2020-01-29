// https://github.com/mikolalysenko/l1-path-finder

// https://en.wikipedia.org/wiki/Jump_point_search
use std::collections::BinaryHeap;

use std::cmp::Ordering;
use std::f32::consts::SQRT_2;
use std::f32::EPSILON;
use std::ops::Sub;

use ndarray::Array;
use ndarray::Array2;

use fnv::FnvHashMap;
// use fnv::FnvHasher;

#[allow(dead_code)]
fn absdiff<T>(x: T, y: T) -> T
where
    T: Sub<Output = T> + PartialOrd,
{
    if x < y {
        y - x
    } else {
        x - y
    }
}

fn manhattan_heuristic(source: &Point2d, target: &Point2d) -> f32 {
    (absdiff(source.x, target.x) + absdiff(source.y, target.y)) as f32
}

static SQRT_2_MINUS_2: f32 = SQRT_2 - 2.0;

fn octal_heuristic(source: &Point2d, target: &Point2d) -> f32 {
    let dx = absdiff(source.x, target.x);
    let dy = absdiff(source.y, target.y);
    let min = std::cmp::min(dx, dy);
    dx as f32 + dy as f32 + SQRT_2_MINUS_2 * min as f32
}

fn euclidean_heuristic(source: &Point2d, target: &Point2d) -> f32 {
    let x = source.x as i32 - target.x as i32;
    let xx = x * x;
    let y = source.y as i32 - target.y as i32;
    let yy = y * y;
    let sum = xx + yy;
    (sum as f32).sqrt()
}

//fn no_heuristic(_source: &Point2d, _target: &Point2d) -> f32 {
//    0.0
//}

/// A direction that is required to be given to a jump point, so that the traversal function knows in which direction it needs to traverse
#[derive(Debug, Copy, Clone, PartialEq)]
struct Direction {
    x: i32,
    y: i32,
}

impl Direction {
    /// Returns true if a direction is in diagonal direction
    fn is_diagonal(self) -> bool {
        match self {
            // Non diagonal movement
            Direction { x: 0, y: 1 }
            | Direction { x: 1, y: 0 }
            | Direction { x: -1, y: 0 }
            | Direction { x: 0, y: -1 } => false,
            _ => true,
        }
    }

    /// Returns direction for 90 degree left turns
    fn left(self) -> Direction {
        match (self.x, self.y) {
            (1, 0) => Direction { x: 0, y: 1 },
            (0, 1) => Direction { x: -1, y: 0 },
            (-1, 0) => Direction { x: 0, y: -1 },
            (0, -1) => Direction { x: 1, y: 0 },
            // Diagonal
            (1, 1) => Direction { x: -1, y: 1 },
            (-1, 1) => Direction { x: -1, y: -1 },
            (-1, -1) => Direction { x: 1, y: -1 },
            (1, -1) => Direction { x: 1, y: 1 },
            _ => panic!("This shouldnt happen"),
        }
    }

    /// Returns direction for 90 degree right turns
    fn right(self) -> Direction {
        match (self.x, self.y) {
            (1, 0) => Direction { x: 0, y: -1 },
            (0, 1) => Direction { x: 1, y: 0 },
            (-1, 0) => Direction { x: 0, y: 1 },
            (0, -1) => Direction { x: -1, y: 0 },
            // Diagonal
            (1, 1) => Direction { x: 1, y: -1 },
            (-1, 1) => Direction { x: 1, y: 1 },
            (-1, -1) => Direction { x: -1, y: 1 },
            (1, -1) => Direction { x: -1, y: -1 },
            _ => panic!("This shouldnt happen"),
        }
    }

    /// Returns direction for 45 degree left turns
    fn half_left(self) -> Direction {
        match (self.x, self.y) {
            (1, 0) => Direction { x: 1, y: 1 },
            (0, 1) => Direction { x: -1, y: 1 },
            (-1, 0) => Direction { x: -1, y: -1 },
            (0, -1) => Direction { x: 1, y: -1 },
            // Diagonal
            (1, 1) => Direction { x: 0, y: 1 },
            (-1, 1) => Direction { x: -1, y: 0 },
            (-1, -1) => Direction { x: 0, y: -1 },
            (1, -1) => Direction { x: 1, y: 0 },
            _ => panic!("This shouldnt happen"),
        }
    }

    /// Returns direction for 45 degree right turns
    fn half_right(self) -> Direction {
        match (self.x, self.y) {
            (1, 0) => Direction { x: 1, y: -1 },
            (0, 1) => Direction { x: 1, y: 1 },
            (-1, 0) => Direction { x: -1, y: 1 },
            (0, -1) => Direction { x: -1, y: -1 },
            // Diagonal
            (1, 1) => Direction { x: 1, y: 0 },
            (-1, 1) => Direction { x: 0, y: 1 },
            (-1, -1) => Direction { x: -1, y: 0 },
            (1, -1) => Direction { x: 0, y: -1 },
            _ => panic!("This shouldnt happen"),
        }
    }

    /// Returns direction for 135 degree left turns
    fn left135(self) -> Direction {
        match (self.x, self.y) {
            // Diagonal
            (1, 1) => Direction { x: -1, y: 0 },
            (-1, 1) => Direction { x: 0, y: -1 },
            (-1, -1) => Direction { x: 1, y: 0 },
            (1, -1) => Direction { x: 0, y: 1 },
            _ => panic!("This shouldnt happen"),
        }
    }

    /// Returns direction for 135 degree right turns
    fn right135(self) -> Direction {
        match (self.x, self.y) {
            // Diagonal
            (1, 1) => Direction { x: 0, y: -1 },
            (-1, 1) => Direction { x: 1, y: 0 },
            (-1, -1) => Direction { x: 0, y: 1 },
            (1, -1) => Direction { x: -1, y: 0 },
            _ => panic!("This shouldnt happen"),
        }
    }
}

/// A struct for saving coordinates
#[derive(Debug, Hash, Eq, PartialEq, Copy, Clone)]
pub struct Point2d {
    x: usize,
    y: usize,
}

impl Point2d {
    pub fn new(x: usize, y: usize) -> Self {
        Point2d { x, y }
    }

    pub fn unpack(&self) -> (usize, usize) {
        (self.x, self.y)
    }

    /// Helper function for quickly summing a point with a direction
    fn add_direction(&self, other: Direction) -> Point2d {
        Point2d {
            x: (self.x as i32 + other.x) as usize,
            y: (self.y as i32 + other.y) as usize,
        }
    }

    /// Returns the direction from self to target point
    fn get_direction(&self, target: &Point2d) -> Direction {
        let x: i32;
        let y: i32;
        match self.x.cmp(&target.x) {
            Ordering::Greater => x = -1,
            Ordering::Less => x = 1,
            Ordering::Equal => x = 0,
        }
        match self.y.cmp(&target.y) {
            Ordering::Greater => y = -1,
            Ordering::Less => y = 1,
            Ordering::Equal => y = 0,
        }
        Direction { x, y }
    }
}

/// A jump point which contains information about the start location, direction it should traverse towards, cost to start, total cost
#[derive(Debug)]
struct JumpPoint {
    start: Point2d,
    direction: Direction,
    cost_to_start: f32,
    total_cost_estimate: f32,
}

/// A comparison method to compare two f32 numbers
impl PartialEq for JumpPoint {
    fn eq(&self, other: &Self) -> bool {
        absdiff(self.total_cost_estimate, other.total_cost_estimate) < EPSILON
    }
}

/// A comparison method to compare two f32 numbers
impl PartialOrd for JumpPoint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        other
            .total_cost_estimate
            .partial_cmp(&self.total_cost_estimate)
    }
}

/// The result of this implementation doesnt seem to matter - instead what matters, is that it is implemented at all
impl Ord for JumpPoint {
    fn cmp(&self, other: &Self) -> Ordering {
        other
            .total_cost_estimate
            .partial_cmp(&self.total_cost_estimate)
            .unwrap()
    }
}

/// Same as Ord
impl Eq for JumpPoint {}

/// The pathfinder struct which can calculate paths on the grid
pub struct PathFinder {
    /// Pathfinder assumes that the borders of the grid are walls.
    /// [
    ///     [0, 0, 0, 0],
    ///     [0, 1, 1, 0],
    ///     [0, 1, 1, 0],
    ///     [0, 0, 0, 0],
    /// ]
    grid: Array2<u8>,
    /// May be 'manhattan', 'octal' or 'euclidean'. 'octal' should probably be the best choice here
    heuristic: String,
    jump_points: BinaryHeap<JumpPoint>,
    /// Contains points which were already visited to construct the path once the goal is found
    came_from: FnvHashMap<Point2d, Point2d>,
}

impl PathFinder {
    /// Returns the answer to 'already visited?', if not visited it adds it to the dictionary
    fn add_came_from(&mut self, p1: &Point2d, p2: &Point2d) -> bool {
        if !self.came_from.contains_key(p1) {
            self.came_from.insert(*p1, *p2);
            return false;
        }
        true
    }

    /// Checks if the point is in the grid. Careful, the 2d grid needs to be similar to numpy arrays, so row major. Grid[y][x]
    fn is_in_grid_bounds(&self, point: &Point2d) -> bool {
        let dim = self.grid.raw_dim();
        let (height, width) = (dim[0], dim[1]);
        // No need to test for 0 <= point.x since usize is used
        return point.x < width && point.y < height;
    }

    /// Checks if the point is in the grid. Careful, the 2d grid needs to be similar to numpy arrays, so row major. Grid[y][x]
    fn is_in_grid(&self, point: &Point2d) -> bool {
        self.grid[[point.y, point.x]] == 1
    }

    /// Returns an option of a Point2d if the point in that direction is not a wall.
    fn new_point_in_grid(&self, point: &Point2d, direction: Direction) -> Option<Point2d> {
        let new_point = point.add_direction(direction);
        if self.is_in_grid(&new_point) {
            return Some(new_point);
        }
        None
    }

    /// Checks if the target point has already been visited.
    fn goal_reached(&self, target: &Point2d) -> bool {
        self.came_from.contains_key(&target)
    }

    /// Construct the path from the came_from hashmap
    /// 1)
    ///     The path can be constructed as minimal as possible (leaving gaps in the path), e.g.
    ///     [
    ///         (0, 0), // Diagonal movement to top right
    ///         (5, 5), // Vertical movement to the top
    ///         (5, 7)
    ///     ]
    /// 2)
    ///     The path can be constructed as fully, e.g.
    ///     [
    ///         (0, 0),
    ///         (1, 1),
    ///         (2, 2),
    ///         (3, 3),
    ///         (4, 4),
    ///         (5, 5),
    ///         (5, 6),
    ///         (5, 7)
    ///     ]
    fn construct_path(
        &self,
        source: &Point2d,
        target: &Point2d,
        construct_full_path: bool,
    ) -> Vec<Point2d> {
        if construct_full_path {
            let mut path: Vec<Point2d> = Vec::with_capacity(100);
            let mut pos = *target;
            path.push(pos);
            while &pos != source {
                let temp_target = *self.came_from.get(&pos).unwrap();
                let dir = pos.get_direction(&temp_target);
                let mut temp_pos = pos.add_direction(dir);
                while temp_pos != temp_target {
                    path.push(temp_pos);
                    temp_pos = temp_pos.add_direction(dir);
                }
                pos = temp_target;
            }
            path.push(*source);
            path.reverse();
            path
        } else {
            let mut path: Vec<Point2d> = Vec::with_capacity(20);
            path.push(*target);
            let mut pos = self.came_from.get(target).unwrap();
            while pos != source {
                pos = self.came_from.get(&pos).unwrap();
                path.push(*pos);
            }
            path.reverse();
            path
        }
    }

    /// The find path algorithm which creates 8 starting jump points around the start point, then only traverses those
    pub fn find_path(&mut self, source: &Point2d, target: &Point2d) -> Vec<Point2d> {
        if !self.is_in_grid_bounds(source) {
            println!(
                "Returning early, source position is not within grid bounds: {:?}",
                source
            );
            return vec![];
        }

        if !self.is_in_grid_bounds(target) {
            println!(
                "Returning early, target position is not within grid bounds: {:?}",
                target
            );
            return vec![];
        }

        if !self.is_in_grid(&source) {
            println!(
                "Returning early, source position is at the position of a wall: {:?}",
                source
            );
            return vec![];
        }
        if !self.is_in_grid(&target) {
            println!(
                "Returning early, target position is at the position of a wall: {:?}",
                target
            );
            return vec![];
        }

        let heuristic: fn(&Point2d, &Point2d) -> f32;
        match self.heuristic.as_ref() {
            "manhattan" => heuristic = manhattan_heuristic,
            "octal" => heuristic = octal_heuristic,
            "euclidean" => heuristic = euclidean_heuristic,
            // Memory overflow!
            // "none" => heuristic = no_heuristic,
            _ => heuristic = octal_heuristic,
        }

        // Clear from last run
        self.jump_points.clear();
        self.came_from.clear();

        // Add 4 starting nodes (diagonal traversals) around source point
        for dir in [
            Direction { x: 1, y: 1 },
            Direction { x: -1, y: 1 },
            Direction { x: -1, y: -1 },
            Direction { x: 1, y: -1 },
        ]
        .iter()
        {
            self.jump_points.push(JumpPoint {
                start: *source,
                direction: *dir,
                cost_to_start: 0.0,
                total_cost_estimate: 0.0 + heuristic(&source, target),
            });
        }

        while let Some(JumpPoint {
            start,
            direction,
            cost_to_start,
            ..
        }) = self.jump_points.pop()
        {
            if self.goal_reached(&target) {
                return self.construct_path(source, target, false);
            }

            self.traverse(&start, direction, &target, cost_to_start, heuristic);
        }

        vec![]
    }

    /// Traverse in a direction (diagonal, horizontal, vertical) until a wall is reached.
    /// If a jump point is encountered (after a wall on left or right side), add it to binary heap of jump points.
    /// If the traversal is diagonal, always split off 45 degree left and right turns and traverse them immediately without adding them to binary heap beforehand.
    fn traverse(
        &mut self,
        start: &Point2d,
        direction: Direction,
        target: &Point2d,
        cost_to_start: f32,
        heuristic: fn(&Point2d, &Point2d) -> f32,
    ) {
        // How far we moved from the start of the function call
        let mut traversed_count: u32 = 0;
        let add_nodes: Vec<(Direction, Direction)> = if direction.is_diagonal() {
            // The first two entries will be checked for left_blocked and right_blocked, if a wall was encountered but that position is now free (forced neighbors?)
            // If the vec has more than 2 elements, then the remaining will not be checked for walls (this is the case in diagonal movement where it forks off to horizontal+vertical movement)
            // (blocked_direction from current_node, traversal_direction)
            let (half_left, half_right) = (direction.half_left(), direction.half_right());
            vec![
                (direction.left135(), direction.left()),
                (direction.right135(), direction.right()),
                (half_left, half_left),
                (half_right, half_right),
            ]
        } else {
            vec![
                (direction.left(), direction.half_left()),
                (direction.right(), direction.half_right()),
            ]
        };
        let mut current_point = *start;
        // Stores wall status - if a side is no longer blocked: create jump point and fork path
        let (mut left_blocked, mut right_blocked) = (false, false);
        loop {
            // Goal found, construct path
            if current_point == *target {
                self.add_came_from(&current_point, &start);
                //                println!("Found goal: {:?} {:?}", current_point, direction);
                //                println!("Size of open list: {:?}", self.jump_points.len());
                //                println!("Size of came from: {:?}", self.came_from.len());
                return;
            }
            // We loop over each direction that isnt the traversal direction
            // For diagonal traversal this is 2 checks (left is wall, right is wall), and 2 forks (horizontal+vertical movement)
            // For non-diagonal traversal this is only checking if there are walls on the side
            for (index, (check_dir, traversal_dir)) in add_nodes.iter().enumerate() {
                // Check if in that direction is a wall
                let check_point_is_in_grid =
                    self.is_in_grid(&current_point.add_direction(*check_dir));

                if (index == 0 && left_blocked || index == 1 && right_blocked || index > 1)
                    && traversed_count != 0
                    && check_point_is_in_grid
                {
                    // If there is no longer a wall in that direction, add jump point to binary heap
                    let new_cost_to_start = if traversal_dir.is_diagonal() {
                        cost_to_start + SQRT_2 * traversed_count as f32
                    } else {
                        cost_to_start + traversed_count as f32
                    };

                    if index < 2 {
                        if self.add_came_from(&current_point, &start) {
                            // We were already at this point because a new jump point was created here - this means we either are going in a circle or we come from a path that is longer?
                            break;
                        }
                        // Add forced neighbor to min-heap
                        self.jump_points.push(JumpPoint {
                            start: current_point,
                            direction: *traversal_dir,
                            cost_to_start: new_cost_to_start,
                            total_cost_estimate: new_cost_to_start
                                + heuristic(&current_point, target),
                        });

                        // Mark the side no longer as blocked
                        if index == 0 {
                            left_blocked = false;
                        } else {
                            right_blocked = false;
                        }
                    // If this is non-diagonal traversal, this is used to store a 'came_from' point
                    } else {
                        // If this is diagonal traversal, instantly traverse the non-diagonal directions without adding them to min-heap first
                        self.traverse(
                            &current_point,
                            *traversal_dir,
                            target,
                            new_cost_to_start,
                            heuristic,
                        );
                        // The non-diagonal traversal created a jump point and added it to the min-heap, so to backtrack from target/goal, we need to add this position to 'came_from'
                        self.add_came_from(&current_point, &start);
                    }
                } else if index == 0 && !check_point_is_in_grid {
                    // If this direction (left) has now a wall, mark as blocked
                    left_blocked = true;
                } else if index == 1 && !check_point_is_in_grid {
                    // If this direction (right) has now a wall, mark as blocked
                    right_blocked = true
                }
            }

            current_point = current_point.add_direction(direction);
            if !self.is_in_grid(&current_point) {
                // Next traversal point is a wall - this traversal is done
                break;
            }
            // Next traversal point is pathable
            traversed_count += 1;
        }
    }

    /// Quickly create new pathfinder
    pub fn new(grid: Array2<u8>, heuristic: &String) -> Self {
        PathFinder {
            grid,
            heuristic: heuristic.clone(),
            jump_points: BinaryHeap::with_capacity(50),
            came_from: FnvHashMap::default(),
        }
    }

    /// A helper function to set up a square grid, border of the grid is set to 0, other values are set to 1
    pub fn create_square_grid(size: usize) -> Array2<u8> {
        // https://stackoverflow.com/a/59043086/10882657
        let mut ndarray = Array2::<u8>::ones((size, size));
        // Set boundaries
        for y in 0..size {
            ndarray[[y, 0]] = 0;
            ndarray[[y, size - 1]] = 0;
        }
        for x in 0..size {
            ndarray[[0, x]] = 0;
            ndarray[[size - 1, x]] = 0;
        }
        ndarray
    }
}

/// A helper function to run the pathfinding algorithm
pub fn jps_test(pf: &mut PathFinder, source: &Point2d, target: &Point2d) -> Vec<Point2d> {
    pf.find_path(&source, &target)
}


use std::fs::File;
use std::io::Read;

/// A helper function to read the grid from file
pub fn read_grid_from_file(path: String) -> Result<(Array2<u8>, u32, u32), std::io::Error> {
    let mut file = File::open(path).unwrap();
    let mut data = String::new();

    file.read_to_string(&mut data).unwrap();
    let mut height = 0;
    let mut width = 0;
    // Create one dimensional vec
    let mut my_vec = Vec::new();
    for line in data.lines() {
        width = line.len();
        height += 1;
        for char in line.chars() {
            my_vec.push(char as u8 - 48);
        }
    }

    let array = Array::from(my_vec).into_shape((height, width)).unwrap();
    Ok((array, height as u32, width as u32))
}

#[cfg(test)] // Only compiles when running tests
mod tests {
    use super::*;
    #[allow(unused_imports)]
    use test::Bencher;

    #[bench]
    fn bench_jps_test_from_file(b: &mut Bencher) {
        // Setup
        let result = read_grid_from_file(String::from("AutomatonLE.txt"));
        let (array, _height, _width) = result.unwrap();
        // Spawn to spawn
        let source = Point2d { x: 32, y: 51 };
        let target = Point2d { x: 150, y: 129 };

        // Main ramp to main ramp
        // let source = Point2d { x: 32, y: 51 };
        // let target = Point2d { x: 150, y: 129 };
        let mut pf = PathFinder::new(array, &String::from("octal"));
        let path = jps_test(&mut pf, &source, &target);
        assert_ne!(0, path.len());
        // Run bench
        b.iter(|| jps_test(&mut pf, &source, &target));
    }

    #[bench]
    fn bench_jps_test_from_file_no_path(b: &mut Bencher) {
        // Setup
        let result = read_grid_from_file(String::from("AutomatonLE.txt"));
        let (mut array, _height, _width) = result.unwrap();
        // Spawn to spawn
        let source = Point2d { x: 32, y: 51 };
        let target = Point2d { x: 150, y: 129 };

        // Block entrance to main base
        for x in 145..=150 {
            for y in 129..=135 {
                array[[y, x]] = 0;
            }
        }

        let mut pf = PathFinder::new(array, &String::from("octal"));
        let path = jps_test(&mut pf, &source, &target);
        assert_eq!(0, path.len());
        // Run bench
        b.iter(|| jps_test(&mut pf, &source, &target));
    }

    #[bench]
    fn bench_jps_test_out_of_bounds1(b: &mut Bencher) {
        let grid = PathFinder::create_square_grid(30);
        let mut pf = PathFinder::new(grid, &String::from("octal"));
        let source: Point2d = Point2d { x: 500, y: 5 };
        let target: Point2d = Point2d { x: 10, y: 12 };
        let path = jps_test(&mut pf, &source, &target);
        assert_eq!(0, path.len());
        b.iter(|| jps_test(&mut pf, &source, &target));
    }

    #[bench]
    fn bench_jps_test_out_of_bounds2(b: &mut Bencher) {
        let grid = PathFinder::create_square_grid(30);
        let mut pf = PathFinder::new(grid, &String::from("octal"));
        let source: Point2d = Point2d { x: 5, y: 5 };
        let target: Point2d = Point2d { x: 500, y: 12 };
        let path = jps_test(&mut pf, &source, &target);
        assert_eq!(0, path.len());
        b.iter(|| jps_test(&mut pf, &source, &target));
    }

    #[bench]
    fn bench_jps_test(b: &mut Bencher) {
        let grid = PathFinder::create_square_grid(30);
        let mut pf = PathFinder::new(grid, &String::from("octal"));
        let source: Point2d = Point2d { x: 5, y: 5 };
        let target: Point2d = Point2d { x: 10, y: 12 };
        b.iter(|| jps_test(&mut pf, &source, &target));
    }
}
