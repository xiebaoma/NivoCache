// Typed key/value usage of NivoCache: POD structs stored in place, structs with
// strings and vectors stored through serialize(), and several typed caches
// sharing one Cache.
#include <nivo/nivo.h>

#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// Trivially copyable: stored as raw bytes and read in place (zero copy).
struct Student {
  int age;
  char name[32];
  double score;
};

// Owns heap memory: lists its fields once and NivoCache handles the encoding.
struct Course {
  std::string title;
  std::vector<int> student_ids;

  template <class Archive>
  void serialize(Archive& archive) {
    archive(title, student_ids);
  }
};

int main() {
  try {
    // 1. POD struct values.
    nivo::NivoCache<int, Student> students(64 * 1024 * 1024);
    students.put(10001, Student{20, "Alice", 95.5});
    nivo::TypedReadHandle<Student> alice = students.get(10001);
    if (!alice) return 1;
    std::cout << alice->name << " age " << alice->age << " score " << alice->score << '\n';

    // 2. Scalars, strings and vectors work out of the box.
    nivo::NivoCache<int, int> counters;
    counters.put(1, 42);
    nivo::NivoCache<std::string, Student> by_name;
    by_name.put("alice", *alice);
    nivo::NivoCache<std::uint64_t, std::vector<int>> grades;
    grades.put(10001, std::vector<int>{90, 95, 88});
    std::optional<std::vector<int>> alice_grades = grades.get_value(10001);
    if (!alice_grades) return 1;
    std::cout << "counter " << *counters.get(1) << ", " << alice_grades->size()
              << " grades, by name " << by_name.get("alice")->name << '\n';

    // 3. Struct values with strings and vectors, via Course::serialize.
    nivo::NivoCache<std::string, Course> courses;
    courses.put("cs101", Course{"Intro to Systems", {10001, 10002}});
    nivo::TypedReadHandle<Course> course = courses.get("cs101");
    if (!course) return 1;
    std::cout << course->title << " has " << course->student_ids.size() << " students\n";

    // 4. Misses, insert-only writes and TTL.
    if (students.get(99999)) return 1;
    if (students.insert(10001, *alice)) return 1;  // key already present
    students.put(10002, Student{19, "Bob", 88.0}, std::chrono::seconds(30));

    // 5. Compare-and-exchange against the version a handle observed. The old
    //    handle stays a valid snapshot after the update.
    nivo::Result<nivo::Version> updated = students.write(10001, Student{21, "Alice", 97.0},
        {.mode = nivo::WriteMode::compare_exchange, .expected_version = alice.version()});
    if (!updated) return 1;
    std::cout << "updated to version " << updated.value() << ", old snapshot still reads age "
              << alice->age << '\n';

    // 6. Several typed caches sharing one Cache, one pool per value type.
    std::unique_ptr<nivo::Cache> shared = nivo::Cache::create().value();
    const nivo::PoolId student_pool =
        shared->add_pool({.name = "students", .capacity_bytes = 8 * 1024 * 1024}).value();
    const nivo::PoolId course_pool =
        shared->add_pool({.name = "courses", .capacity_bytes = 8 * 1024 * 1024}).value();
    nivo::TypedCache<int, Student> shared_students(*shared, student_pool);
    nivo::TypedCache<std::string, Course> shared_courses(*shared, course_pool);
    shared_students.put(10001, *alice);
    shared_courses.put("cs101", *course);
    if (!shared_students.contains(10001) || !shared_courses.contains("cs101")) return 1;

    return 0;
  } catch (const nivo::CacheError& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
