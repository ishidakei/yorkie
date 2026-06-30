#[global_allocator]
static GLOBAL: mimalloc::MiMalloc = mimalloc::MiMalloc;

fn main() {
    std::thread::Builder::new()
        .stack_size(yorkie::stack_size::STACK_SIZE)
        .spawn(|| {
            yorkie::usi::cmd_loop();
        })
        .unwrap()
        .join()
        .unwrap();
}
