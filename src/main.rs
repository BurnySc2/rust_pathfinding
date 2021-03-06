#![feature(cell_update)]
// Testing and benchmark crate
#![feature(test)]
extern crate test;

use std::fs::File;
use std::io::prelude::*;

pub mod algorithms;

fn write_path_to_file(path: Vec<algorithms::jps::Point2d>) {
    let mut file = File::create("path.txt").unwrap();
    for i in path.iter() {
        //        let algorithms::jps::Point2d { x, y } = i;
        let (x, y) = i.unpack();
        file.write_fmt(format_args!("{},{}\n", x, y));
    }
}

fn main() {
    //    // Test on actual map AutomatonLE.txt
    let result = algorithms::jps::read_grid_from_file(String::from("AutomatonLE.txt"));
    let (array, _height, _width) = result.unwrap();
    // A simple short path example around a corner or two
    //        let source = algorithms::jps::Point2d { x: 70, y: 100 };
    //        let target = algorithms::jps::Point2d { x: 100, y: 114 };

    // Spawn to spawn
    let source = algorithms::jps::Point2d::new(29, 65);
    let target = algorithms::jps::Point2d::new(154, 114);

    // Main ramp to main ramp
    //    let source = algorithms::jps::Point2d { x: 32, y: 51 };
    //    let target = algorithms::jps::Point2d { x: 150, y: 129 };
    let mut pf = algorithms::jps::PathFinder::new(array, &String::from("octal"));
    let path = algorithms::jps::jps_test(&mut pf, &source, &target, false);
    println!("Path: {:?}", path);

    // Test on empty 100x100 grid
    //            let source = algorithms::jps::Point2d { x: 5, y: 5 };
    //            let target = algorithms::jps::Point2d { x: 10, y: 12 };
    //            let grid = algorithms::jps::grid_setup(15);
    //            let path = algorithms::jps::jps_test(grid, source, target);
    //            println!("Path: {:?}", path);
    if path.len() > 0 {
        write_path_to_file(path);
    }
}
