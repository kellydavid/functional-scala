// Copyright(C) 2018-2019 John A. De Goes. All rights reserved.

package net.degoes.essentials

import net.degoes.essentials.tc_motivating.LessThan

object types {
  type ??? = Nothing


  // Colours = {x | x is a colour}
  // { p : p is a person in the room without black hair }
  // Colours = { x : x is a colour }

  val int: Int = 42 // not how colon is a type description operation same as set builder notation above

  //
  // EXERCISE 1
  //
  // List all values of the type `Unit`.
  //
  val UnitValues: Set[Unit] = Set(())

  // empty parameter list and unit have same syntax
  // type Unit is Unit . Value Unit is ()

  //
  // EXERCISE 2
  //
  // List all values of the type `Nothing`.
  //
  val NothingValues: Set[Nothing] = Set.empty

  // Nothing = { }
  // val example: Nothing = <cannot put anything here>
//  val example: Nothing = ???
//  val example2: Nothing = (throw ex) // act of throwing scala assigns value of nothing
  // lazy val example: Nothing = example // will cause stack overflow

  //
  // EXERCISE 3
  //
  // List all values of the type `Boolean`.
  //
  val BoolValues: Set[Boolean] = Set(true, false)

  //
  // EXERCISE 4
  //
  // List all values of the type `Either[Unit, Boolean]`.
  //
  val EitherUnitBoolValues: Set[Either[Unit, Boolean]] = Set(Left(()), Right(false), Right(true))

  //
  // EXERCISE 5
  //
  // List all values of the type `(Boolean, Boolean)`.
  //
  val TupleBoolBoolValues: Set[(Boolean, Boolean)] =
  Set(
    (false, false),
    (false, true),
    (true, false),
    (true, true)
  )

  //
  // EXERCISE 6
  //
  // List all values of the type `Either[Either[Unit, Unit], Unit]`.
  //
  val EitherEitherUnitUnitUnitValues: Set[Either[Either[Unit, Unit], Unit]] =
  Set(
    Left(Left(())),
    Left(Right(())),
    Right(())
  )

  //
  // EXERCISE 7
  //
  // Given:
  // A = { true, false }
  // B = { "red", "green", "blue" }
  //
  // List all the elements in `A * B`.
  //
  val AProductB: Set[(Boolean, String)] = {
    val A = Set(true, false)
    val B = Set("red", "green", "blue")
    for {
      a <- A
      b <- B
    } yield (a, b)
  }

  //
  // EXERCISE 8
  //
  // Given:
  // A = { true, false }
  // B = { "red", "green", "blue" }
  //
  // List all the elements in `A + B`.
  //
  val ASumB: Set[Either[Boolean, String]] =
  Set(
    Left(true),
    Left(false),
    Right("red"),
    Right("green"),
    Right("blue")
  )

  val ASumB2: Set[Either[Boolean, String]] =
    Set(true, false).map(Left(_)) ++ Set("red", "green", "blue").map(Right(_))

// /  sealed trait Either[A, B]
//  case class Left[A, B](value: A) extends Either[A, B]
//  case class Right[A, B](value: B) extends Either[A, B]

  sealed trait JobTitle // this will be an n-way sum type
  // case classes or case objects or sealed traits can be in sum type
  // size of sum set is not connected to number of terms inside set.
  // minimum size of n-way set is n
  object Manager extends JobTitle
  object Peon extends JobTitle
  object Programmer extends JobTitle
  case class SalesPerson(level: Int) extends JobTitle

  //
  // EXERCISE 9
  //
  // Create a product type of `Int` and `String`, representing the age and
  // name of a person.
  //
  type Person1 = (Int, String)
  final case class Person2(age: Int, name: String)

  //
  // EXERCISE 10
  //
  // Prove that `A * 1` is equivalent to `A` by implementing the following two
  // functions.
  //
  def to1[A](t: (A, Unit)): A = ???
  def from1[A](a: A): (A, Unit) = ???

  //
  // EXERCISE 11
  //
  // Prove that `A * 0` is equivalent to `0` by implementing the following two
  // functions.
  //
  // Nothing is a sub type of every other type which allows you to use anything
  def to2[A](t: (A, Nothing)): Nothing = t._2
  def from2[A](n: Nothing): (A, Nothing) = (n, n)

  def magicalConversion[A](n: Nothing): A = n // no values of nothing exist so this is ok

  //
  // EXERCISE 12
  //
  // Create a sum type of `Int` and `String` representing the identifier of
  // a robot (a number) or the identifier of a person (a name).
  //
  type Identifier1 = (Int, String)
  sealed trait Identifier2
  case class RobotIdentifier(id: Int) extends Identifier2
  case class PersonIdentifier(name: String) extends Identifier2

  //
  // EXERCISE 13
  //
  // Prove that `A + 0` is equivalent to `A` by implementing the following two
  // functions.
  //
  def to3[A](t: Either[A, Nothing]): A = t.right.get
  def from3[A](a: A): Either[A, Nothing] = Left(a)

  //
  // EXERCISE 14
  //
  // Create either a sum type or a product type (as appropriate) to represent a
  // credit card, which has a number, an expiration date, and a security code.
  //
//  type CreditCard = ???
  case class CreditCard(number: Int, expDate: String, secCode: Int)
  // product type becuase has to have all three definied at same time

  //
  // EXERCISE 15
  //
  // Create either a sum type or a product type (as appropriate) to represent a
  // payment method, which could be a credit card, bank account, or
  // cryptocurrency.
  //
//  type PaymentMethod = ???
  sealed trait PaymentMethod
  case object BankAccount extends PaymentMethod
  case object CryptoCurrency extends PaymentMethod
  // ^ exclusive payment method types

  //
  // EXERCISE 16
  //
  // Create either a sum type or a product type (as appropriate) to represent an
  // employee at a company, which has a title, salary, name, and employment date.
  //
//  type Employee = ???

  sealed trait Company
  case object Bigdata extends Company
  case class Employee(company: Company, title: String, salary: Int, emplymentDate: String)

  //
  // EXERCISE 17
  //
  // Create either a sum type or a product type (as appropriate) to represent a
  // piece on a chess board, which could be a pawn, rook, bishop, knight,
  // queen, or king.
  //
  type ChessPiece = ???

  // def construct(value: String): Email = ???

  //
  // EXERCISE 18
  //
  // Create a "smart constructor" for `Programmer` that only permits levels
  // that are non-negative.
  //
//  final case class Programmer private (level: Int)
//  object Programmer {
//    def apply(level: Int): Option[Programmer] =
//      ???
//  }

  //
  // EXERCISE 19
  // 
  // Using algebraic data types and smart constructors, make it impossible to
  // construct a `BankAccount` with an illegal (undefined) state in the 
  // business domain. Note any limitations in your solution.
  //
  case class BankAccount(
    ownerId: String, 
    balance: BigDecimal,
    accountType: String, 
    openedDate: Long)

  //
  // EXERCISE 20 
  //
  // Create an ADT model of a game world, including a map, a player, non-player
  // characters, different classes of items, and character stats.
  //
//  type GameWorld = ???
  // example of smart constructor
  // business goal: prevent run time representing illigal states that the business has no meaning for
  final case class Programmer2 private(level: Int)
  object Programmer2 {
    def apply(level: Int): Option[Programmer2] =
      if (level < 0) None
      else Some(new Programmer2(level))
  }

//  case class BankAccount (
//                         ownerId: String, // shouldn't be empty string, should it be a guid?
//                         balance: BigDecimal, // maybe the business rule does not allow negative balance to exist
//                         accountType: String, // should be a sum type
//                         openedDate: java.time.Instant
//                         )

  sealed trait AccountType
  case object CheckingAccount extends AccountType
  case object SavingsAccount extends AccountType

  case class Balance(value: BigDecimal)
  object Balance {
    def apply(value: BigDecimal): Option[Balance] =
      if(value < 0) None
      else Some(new Balance(value))
  }

//  case class OwnerId(value: String)
//  object OwnerId {
//    def apply(value: String): Option[OwnerId] =
//      if(value.length)
//  }

//case class BankAccount[A] (
//                         ownerId: A, // shouldn't be empty string, should it be a guid?
//                         balance: Balance, // maybe the business rule does not allow negative balance to exist
//                         accountType: AccountType, // should be a sum type
//                         openedDate: java.time.Instant
//                       )

  // if the owner id is completely run time dependent, ie db manages this, then rip this dependency out of the code
  // use polymorphism to accomplish this
//  def transfer[A](amount: BigDecimal,
//                  acc1: BankAccount[A],
//                  acc2: BankAccount[A]
//                 ) = ???


}

object functions {
  type ??? = Nothing

  // Two Sets:
  // Domain = { a1, a2, ..., an}
  // Codomain = {b1, b2, ... bm}
  // A function `f: Ddmain => Codomain
  // Is a mapping from `Domain` to `Codomain`
  // such that for every `a` in `Domain`
  // `f(a)` is in `Codomain`.

  //
  // EXERCISE 1
  //
  // Convert the following non-function into a function.
  //
  def parseInt1(s: String): Int = s.toInt // this is a lie - not total
  def parseInt2(s: String): Option[Int] = scala.util.Try(s.toInt).toOption

  //
  // EXERCISE 2
  //
  // Convert the following non-function into a function.
  //
  def arrayUpdate1[A](arr: Array[A], i: Int, f: A => A): Unit =
    arr.update(i, f(arr(i))) // 1. can throw exception (i outside boundary of array) - not total
                             // 2. has a side effect
                             // if return type is Unit then implementation should just be ()
//  def arrayUpdate2[A](arr: Array[A], i: Int, f: A => A): Array[A] =
//                               if(i >= 0 && i < arr.size) {
//                                 arr.updated(i, f(arr(i)))
//                               }else arr

  //
  // EXERCISE 3
  //
  // Convert the following non-function into a function.
  //
  def divide1(a: Int, b: Int): Int = a / b //divide by zero - not total
  def divide2(a: Int, b: Int): ??? = ???

  //
  // EXERCISE 4
  //
  // Convert the following non-function into a function.
  //
  var id = 0
  def freshId1(): Int = {
    val newId = id
    id += 1
    newId
  } // has a side effect
  def freshId2(/* ??? */): (Int, Int) = ???

  //
  // EXERCISE 5
  //
  // Convert the following non-function into a function.
  //
  import java.time.LocalDateTime
  def afterOneHour1: LocalDateTime = LocalDateTime.now.plusHours(1) // non deterministic - push now higher (solve the problem at a high level, all lowel level code does not get complex and can be tested.)
  def afterOneHour2(/* ??? */): LocalDateTime = ???

  //
  // EXERCISE 6
  //
  // Convert the following non-function into function.
  //
  def head1[A](as: List[A]): A = {
    if (as.length == 0) println("Oh no, it's impossible!!!")
    as.head
  }
  def head2[A](as: List[A]): ??? = ???

  //
  // EXERCISE 7
  //
  // Convert the following non-function into a function.
  //
  trait Account
  trait Processor {
    def charge(account: Account, amount: Double): Unit
  }
  case class Coffee() {
    val price = 3.14
  }
  def buyCoffee1(processor: Processor, account: Account): Coffee = {
    val coffee = Coffee()
    processor.charge(account, coffee.price)
    coffee
  }
  final case class Charge(account: Account, amount: Double)
  def buyCoffee2(account: Account): (Coffee, Charge) = { //push the processor.charge problem higher
    val coffee = Coffee()
    (coffee, Charge(account, coffee.price))
  }
  //
  // EXERCISE 8
  //
  // Implement the following function under the Scalazzi subset of Scala.
  //
  def printLine(line: String): Unit = ()

  //
  // EXERCISE 9
  //
  // Implement the following function under the Scalazzi subset of Scala.
  //
  def readLine: String = "foo" // type signature tells us that it is a constant

  //
  // EXERCISE 10
  //
  // Implement the following function under the Scalazzi subset of Scala.
  //
  def systemExit(code: Int): Unit = ???

  //
  // EXERCISE 11
  //
  // Rewrite the following non-function `printer1` into a pure function, which
  // could be used by pure or impure code.
  //
  def printer1(): Unit = { // side effects / non pure
    println("Welcome to the help page!")
    println("To list commands, type `commands`.")
    println("For help on a command, type `help <command>`")
    println("To exit the help page, type `exit`.")
  }
  def printer2[A](println: String => A, combine: (A, A) => A): A =
    Seq(
        "Welcome to the help page!",
        "To list commands, type `commands`.",
        "For help on a command, type `help <command>`",
        "To exit the help page, type `exit`.",
    ).map(println).reduce(combine)

  //
  // EXERCISE 12
  //
  // Create a purely-functional drawing library that is equivalent in
  // expressive power to the following procedural library.
  //
  trait Draw {
    def goLeft(): Unit
    def goRight(): Unit
    def goUp(): Unit
    def goDown(): Unit
    def draw(): Unit
    def finish(): List[List[Boolean]]
  }
  def draw1(size: Int): Draw = new Draw {
    val canvas = Array.fill(size, size)(false)
    var x = 0
    var y = 0

    def goLeft(): Unit = x -= 1
    def goRight(): Unit = x += 1
    def goUp(): Unit = y += 1
    def goDown(): Unit = y -= 1
    def draw(): Unit = {
      def wrap(x: Int): Int =
        if (x < 0) (size - 1) + ((x + 1) % size) else x % size

      val x2 = wrap(x)
      val y2 = wrap(y)

      canvas.updated(x2, canvas(x2).updated(y2, true))
    }
    def finish(): List[List[Boolean]] =
      canvas.map(_.toList).toList
  }
  def draw2(size: Int /* ... */): ??? = ???

  case class CanvasState(x: Int, y: Int, bitmap: List[List[Double]])
  type DrawCommand = CanvasState => CanvasState
  // could then pass a list of draw commands to draw()


}

object higher_order {

//  def foo(x: Int): Int = ??? // monomorphic
//  def foo[A](x: Int): Int = ??? // polymorphic function
//
//  case class Person(name: String, age: Age)
//  case class Age(value: Int)

//  String => Age
//  String => Int
//  Int => Age

//  def joinAge(f: String => Int, g: Int => Age): String = (s: String) => g(f(s)) OR Age(42)
//  def compose[]()
//  the function knows too much ^

  //
  // EXERCISE 1
  //
  // Implement the following higher-order function.
  //
  def fanout[A, B, C](f: A => B, g: A => C): A => (B, C) = (a: A) => (f(a), g(a))

  //
  // EXERCISE 2
  //
  // Implement the following higher-order function.
  //
  def cross[A, B, C, D](f: A => B, g: C => D): (A, C) => (B, D) =
    (a: A, c: C) => (f(a), g(c))

  //
  // EXERCISE 3
  //
  // Implement the following higher-order function.
  //
  def either[A, B, C](f: A => B, g: C => B): Either[A, C] => B = ???

  //
  // EXERCISE 4
  //
  // Implement the following higher-order function.
  //
  def choice[A, B, C, D](f: A => B, g: C => D): Either[A, C] => Either[B, D] =
    ???

  //
  // EXERCISE 5
  //
  // Implement the following higher-order function.
  //
  def compose[A, B, C](f: B => C, g: A => B): A => C =
    ???

  //
  // EXERCISE 6
  //
  // Implement the following higher-order function. After you implement
  // the function, interpret its meaning.
  //
//  def alt[E1, E2, A, B](l: Parser[E1, A], r: E1 => Parser[E2, B]): Parser[(E1, E2), Either[A, B]] = {
//    Parser[(E1, E2), Either[A, B]](run = (s: String) => {???}

//  def alt[E1, E2, A, B](l: Parser[E1, A], r: E1 => Parser[E2, B]):
//    Parser[(E1, E2), Either[A, B]] =
//     ???

//  def alt[E1, E2, A, B](l: Parser[E1, A], r: E1 => Parser[E2, B]):
//  Parser[(E1, E2), Either[A, B]] =
//    Parser[(E1, E2), Either[A, B]](
//      (input: String) => (??? : Either[(E1, E2), (String, Either[A, B])])
//    )

  def alt[E1, E2, A, B](l: Parser[E1, A], r: E1 => Parser[E2, B]):
  Parser[(E1, E2), Either[A, B]] =
    Parser[(E1, E2), Either[A, B]](
      (input: String) =>
        l.run(input) match {
          case Left(e1) => r(e1).run(input) match {
            case Left(e2) => Left((e1, e2))
            case Right((in, b)) => Right((in, Right(b)))
          }
          case Right((in, a)) => Right(in, Left(a))
        }
    )

  case class Parser[+E, +A](run: String => Either[E, (String, A)])
  object Parser {
    final def fail[E](e: E): Parser[E, Nothing] = // we know this cannot fail looking at the second type parameter of the return type
      Parser(_ => Left(e))

    final def point[A](a: => A): Parser[Nothing, A] =
      Parser(input => Right((input, a)))

    final def char: Parser[Unit, Char] = // there is only one failure scenario, this is why unit is error type
      Parser(input =>
        if (input.length == 0) Left(())
        else Right((input.drop(1), input.charAt(0))))
  }
}

object poly_functions {
  //
  // EXERCISE 1
  //
  // Create a polymorphic function of two type parameters `A` and `B` called
  // `snd` that returns the second element out of any pair of `A` and `B`.
  //
  object snd {
    def apply[A, B](t: (A, B)): B = ???
  }
  snd((1, "foo")) // "foo"
  snd((true, List(1, 2, 3))) // List(1, 2, 3)

  //
  // EXERCISE 2
  //
  // Create a polymorphic function called `repeat` that can take any
  // function `A => A`, and apply it repeatedly to a starting value
  // `A` the specified number of times.
  //
  object repeat {
    def apply[A](n: Int)(a: A, f: A => A): A =
//      n match {
//
//        case != 0 => apply(n)
//      }

      n match {
        case 0 => a
        case m => apply(m - 1)(f(a), f)
      }
//      Range(0, n).map()
  }
  repeat[   Int](100)( 0, _ +   1) // 100
  repeat[String]( 10)("", _ + "*") // "**********"

  //
  // EXERCISE 3
  //
  // Count the number of unique implementations of the following method.
  //
  def countExample1[A, B](a: A, b: B): Either[A, B] = ???
  val countExample1Answer = ???

  //
  // EXERCISE 4
  //
  // Count the number of unique implementations of the following method.
  //
  def countExample2[A, B](f: A => B, g: A => B, a: A): B =
    ???
  val countExample2Answer = ???

  //
  // EXERCISE 5
  //
  // Implement the function `groupBy1`.
  //
  val Data =
    "poweroutage;2018-09-20;level=20" :: Nil
  val ByDate: String => String =
    (data: String) => data.split(";")(1)
  val Reducer: (String, List[String]) => String =
    (date, events) =>
      "On date " +
        date + ", there were " +
        events.length + " power outages"
  val ExpectedResults =
    Map("2018-09-20" ->
      "On date 2018-09-20, there were 1 power outages")
  def groupBy1(
    l: List[String],
    by: String => String)(
      reducer: (String, List[String]) => String):
      Map[String, String] =
    l.map((s: String) => (ByDate(s), s) ).groupBy(_._1).map(x => (x._1, Reducer(x._1, x._2.map(_._2))))
  groupBy1(Data, ByDate)(Reducer) == ExpectedResults

  //
  // EXERCISE 6
  //
  // Make the function `groupBy1` as polymorphic as possible and implement
  // the polymorphic function. Compare to the original.
  //
  object groupBy2 {

    def apply[A, B, C](
                  events: List[A],
                  by: A => B)(
                  reducer: (B, List[A]) => C): Map[B, C] =
//      events.map(e => (by(e), e)).groupBy(_._1).map(x => (x._1, reducer(x._1, x._2)))
    events.groupBy(by).map{ case (b: B, as: List[A]) => (b, reducer(b, as)) }
//      events.groupBy(by).map(case (k,v) => (k, reducer())

  }
  // groupBy2(Data, By)(Reducer) == Expected
}

object higher_kinded {
  // A type is a set of values
  // { 1, "foo", true, List(1, 2, 3) }
  // { Int, Boolean, String, ..., } can have mathematical set of types but not in scala
  // * = { Int, Boolean, String, ..., }
  //   = { x | x is a type in scala }

  // List : * => *
  // List(Int) = List[Int]
  // List(String) = List[String]
  //

  // * => * = { f | f is a type constructor that takes 1 type}
  //        = { List, Set, Option, Future, Try, ... }
  //

  // [*, *] => * = { f | f is a type constructor that takes 2 types }
  //
  //             = { Map, Either, Function, Tuple2, .... }
  //

  // f: Int

  //
  // (* => *) => * // higher order kind aka higher kinded type
  //               = { Functor, Monad, ... }
  //
  def foo[A[_], B] = ???
  trait Foo[A[_], B]


//  type Bar = Foo[List, Int] // Wont work as we are passing * as parameter (List)
  type Bar = Foo[List, Int]
  val foo2: (Int, Int) => Int = ???
//  foo2((i: Int) => + 1, 2)

  val myTest: List[Int] = ???

  type ?? = Nothing
  type ???[A] = Nothing
  type ????[A, B] = Nothing
  type ?????[F[_]] = Nothing

  trait `* => *`[F[_]]
  trait `[*, *] => *`[F[_, _]]
  trait `(* => *) => *`[T[_[_]]]

  //
  // EXERCISE 1
  //
  // Identify a type constructor that takes one type parameter of kind `*`
  // (i.e. has kind `* => *`), and place your answer inside the square brackets.
  //
  type Answer1 = `* => *`[???]

  //
  // EXERCISE 2
  //
  // Identify a type constructor that takes two type parameters of kind `*` (i.e.
  // has kind `[*, *] => *`), and place your answer inside the square brackets.
  //
//  type Answer2 = `[*, *] => *` [???]

  //
  // EXERCISE 3
  //
  // Create a new type called `Answer3` that has kind `*`.
  //
  trait Answer3 /*[]*/

  //
  // EXERCISE 4
  //
  // Create a trait with kind `[*, *, *] => *`.
  //
  trait Answer4 /*[]*/

  //
  // EXERCISE 5
  //
  // Create a new type that has kind `(* => *) => *`.
  //
  type NewType1 [A[_[_]]] /* ??? */
//  type Answer5 = `(* => *) => *` ??? //[A[_[_]]]

  //
  // EXERCISE 6
  //
  // Create a trait with kind `[* => *, (* => *) => *] => *`.
  //
  trait Answer6 /*[]*/

  //
  // EXERCISE 7
  //
  // Create an implementation of the trait `CollectionLike` for `List`.
  //
  trait CollectionLike[F[_]] {
    def empty[A]: F[A]

    def cons[A](a: A, as: F[A]): F[A]

    def uncons[A](as: F[A]): Option[(A, F[A])]

    final def singleton[A](a: A): F[A] =
      cons(a, empty[A])

    final def append[A](l: F[A], r: F[A]): F[A] =
      uncons(l) match {
        case Some((l, ls)) => append(ls, cons(l, r))
        case None => r
      }

    final def filter[A](fa: F[A])(f: A => Boolean): F[A] =
      bind(fa)(a => if (f(a)) singleton(a) else empty[A])

    final def bind[A, B](fa: F[A])(f: A => F[B]): F[B] =
      uncons(fa) match {
        case Some((a, as)) => append(f(a), bind(as)(f))
        case None => empty[B]
      }

    final def fmap[A, B](fa: F[A])(f: A => B): F[B] = {
      val single: B => F[B] = singleton[B](_)

      bind(fa)(f andThen single)
    }
  }
  val ListCollectionLike: CollectionLike[List] = ???

  //
  // EXERCISE 8
  //
  // Implement `Sized` for `List`.
  //
  trait Sized[F[_]] {
    // This method will return the number of `A`s inside `fa`.
    def size[A](fa: F[A]): Int
  }
  val ListSized: Sized[List] =
    new Sized[List] {
      def size[A](fa: List[A]): Int = fa.length
    }

  //
  // EXERCISE 9
  //
  // Implement `Sized` for `Map`, partially applied with its first type
  // parameter to `String`.
  //
//  type A = Sized[Map] //
//  val MapStringSized: Sized[Map[String, ?]] = ???
//  new Sized[Map[String, ?]] {
//
//  }

  //
  // EXERCISE 10
  //
  // Implement `Sized` for `Map`, partially applied with its first type
  // parameter to a user-defined type parameter.
  //
  def MapSized2[K]: Sized[Map[K, ?]] =
    ???

  //
  // EXERCISE 11
  //
  // Implement `Sized` for `Tuple3`.
  //
  def Tuple3Sized[C, B]: ?? = ???
}

object tc_motivating {
  /*
  A type class is a tuple of three things:

  1. A set of types and / or type constructors.
  2. A set of operations on values of those types.
  3. A set of laws governing the operations.

  A type class instance is an instance of a type class for a given
  set of types.
  */

//  sealed trait Ordering
//  case object LessThan extends Ordering
//  case object EqualTo extends Ordering
//  case object GreaterThan extends Ordering
//  trait Comparable[A] {
//    def compare(that: A): Ordering
//  }
//
//  case class Person(name: String, age: Int) extends Comparable[Person] {
//    def compare(that: Person): Ordering = {
//      if(this.age < that.age) LessThan
//      EqualTo
//    }
//  }
//
//  def sort[A <: Comparable[A]](list: List[A]): List[A] = ???

  /**
   * All implementations are required to satisfy the transitivityLaw.
   *
   * Transitivity Law:
   * forall a b c.
   *   lt(a, b) && lt(b, c) ==
   *     lt(a, c) || (!lt(a, b) || !lt(b, c))
   */
  trait LessThan[A] {
    def lt(l: A, r: A): Boolean

    final def transitivityLaw(a: A, b: A, c: A): Boolean =
      (lt(a, b) && lt(b, c) == lt(a, c)) ||
      (!lt(a, b) || !lt(b, c))
  }
  implicit class LessThanSyntax[A](l: A) {
    def < (r: A)(implicit A: LessThan[A]): Boolean = A.lt(l, r)
    def >= (r: A)(implicit A: LessThan[A]): Boolean = !A.lt(l, r)
  }
  object LessThan {
    def apply[A](implicit A: LessThan[A]): LessThan[A] = A

    implicit val LessThanInt: LessThan[Int] = new LessThan[Int] {
      def lt(l: Int, r: Int): Boolean = l < r
    }
    implicit def LessThanList[A: LessThan]: LessThan[List[A]] = new LessThan[List[A]] {
      def lt(l: List[A], r: List[A]): Boolean =
        (l, r) match {
          case (Nil, Nil) => false
          case (Nil, _) => true
          case (_, Nil) => false
          case (l :: ls, r :: rs) => l < r && lt(ls, rs)
        }
    }
  }

  def sort[A: LessThan](l: List[A]): List[A] = l match {
    case Nil => Nil
    case x :: xs =>
      val (lessThan, notLessThan) = xs.partition(_ < x)

      sort(lessThan) ++ List(x) ++ sort(notLessThan)
  }

  object oop {
    trait Comparable[A] {
      def lessThan(a: A): Boolean
    }
    def sortOOP[A <: Comparable[A]](l: List[A]): List[A] =
      ???
    case class Person(name: String, age: Int) extends Comparable[Person] {
      def lessThan(a: Person): Boolean = ???
    }
  }

  sort(List(1, 2, 3))
  sort(List(List(1, 2, 3), List(9, 2, 1), List(1, 2, 9)))
}

object hashmap {
  trait Eq[A] {
    def eq(l: A, r: A): Boolean
  }
  object Eq {
    def apply[A](implicit A: Eq[A]): Eq[A] = A

    implicit val EqInt: Eq[Int] =
      new Eq[Int] {
        def eq(l: Int, r: Int): Boolean = l == r
      }
  }
  implicit class EqSyntax[A](l: A) {
    def === (r: A)(implicit A: Eq[A]): Boolean = A.eq(l, r)
  }

  trait Hash[A] extends Eq[A] {
    def hash(a: A): Int

    final def hashConsistencyLaw(a1: A, a2: A): Boolean =
      eq(a1, a2) == ((hash(a1) === hash(a2)) || !eq(a1, a2))
  }
  object Hash {
    def apply[A](implicit A: Hash[A]): Hash[A] = A

    implicit val HashInt: Hash[Int] =
      new Hash[Int] {
        def hash(a: Int): Int = a

        def eq(l: Int, r: Int): Boolean = l == r
      }
  }
  implicit class HashSyntax[A](val a: A) extends AnyVal {
    def hash(implicit A: Hash[A]): Int = A.hash(a)
  }

  case class Person(age: Int, name: String)
  object Person {
    implicit val HashPerson: Hash[Person] = ???
  }

  class HashPerson(val value: Person) extends AnyVal
  object HashPerson {
    implicit val HashHashPerson: Hash[HashPerson] = ???
  }

  class HashMap[K, V] {
    def size: Int = ???

    def insert(k: K, v: V)(implicit K: Hash[K]): HashMap[K, V] = ???
  }
  object HashMap {
    def empty[K, V]: HashMap[K, V] = ???
  }

  Hash[Int].hash(3)

  trait Hashable {
    def hash: Int
  }

  class HashMapOOP[K <: Hashable, V] {
    def size: Int = ???

    def insert(k: K, v: V): HashMap[K, V] = ???
  }
}

object typeclasses {
  /**
   * {{
   * Reflexivity:   a ==> equals(a, a)
   *
   * Transitivity:  equals(a, b) && equals(b, c) ==>
   *                equals(a, c)
   *
   * Symmetry:      equals(a, b) ==> equals(b, a)
   * }}
   */
  trait Eq[A] {
    def equals(l: A, r: A): Boolean
  }
  object Eq {
    def apply[A](implicit eq: Eq[A]): Eq[A] = eq

    implicit val EqInt: Eq[Int] = new Eq[Int] {
      def equals(l: Int, r: Int): Boolean = l == r
    }
    implicit def EqList[A: Eq]: Eq[List[A]] =
      new Eq[List[A]] {
        def equals(l: List[A], r: List[A]): Boolean =
          (l, r) match {
            case (Nil, Nil) => true
            case (Nil, _) => false
            case (_, Nil) => false
            case (l :: ls, r :: rs) =>
              Eq[A].equals(l, r) && equals(ls, rs)
          }
      }
  }
  implicit class EqSyntax[A](val l: A) extends AnyVal {
    def === (r: A)(implicit eq: Eq[A]): Boolean =
      eq.equals(l, r)
  }

  //
  // Scalaz 7 Encoding
  //
  sealed trait Ordering
  case object EQUAL extends Ordering
  case object LT extends Ordering
  case object GT extends Ordering
  object Ordering {
    implicit val OrderingEq: Eq[Ordering] = new Eq[Ordering] {
      def equals(l: Ordering, r: Ordering): Boolean = l == r
    }
  }

  trait Ord[A] {
    def compare(l: A, r: A): Ordering

    final def eq(l: A, r: A): Boolean = compare(l, r) == EQUAL
    final def lt(l: A, r: A): Boolean = compare(l, r) == LT
    final def lte(l: A, r: A): Boolean = lt(l, r) || eq(l, r)
    final def gt(l: A, r: A): Boolean = compare(l, r) == GT
    final def gte(l: A, r: A): Boolean = gt(l, r) || eq(l, r)

    final def transitivityLaw1(a: A, b: A, c: A): Boolean =
      (lt(a, b) && lt(b, c) == lt(a, c)) ||
      (!lt(a, b) || !lt(b, c))

    final def transitivityLaw2(a: A, b: A, c: A): Boolean =
      (gt(a, b) && gt(b, c) == gt(a, c)) ||
      (!gt(a, b) || !gt(b, c))

    final def equalityLaw(a: A, b: A): Boolean =
      (lt(a, b) && gt(a, b) == eq(a, b)) ||
      (!lt(a, b) || !gt(a, b))
  }
  object Ord {
    def apply[A](implicit A: Ord[A]): Ord[A] = A

    implicit val OrdInt: Ord[Int] = new Ord[Int] {
      def compare(l: Int, r: Int): Ordering =
        if (l < r) LT else if (l > r) GT else EQUAL
    }
  }
  implicit class OrdSyntax[A](val l: A) extends AnyVal {
    def =?= (r: A)(implicit A: Ord[A]): Ordering =
      A.compare(l, r)

    def < (r: A)(implicit A: Ord[A]): Boolean =
      Eq[Ordering].equals(A.compare(l, r), LT)

    def <= (r: A)(implicit A: Ord[A]): Boolean =
      (l < r) || (this === r)

    def > (r: A)(implicit A: Ord[A]): Boolean =
      Eq[Ordering].equals(A.compare(l, r), GT)

    def >= (r: A)(implicit A: Ord[A]): Boolean =
      (l > r) || (this === r)

    def === (r: A)(implicit A: Ord[A]): Boolean =
      Eq[Ordering].equals(A.compare(l, r), EQUAL)

    def !== (r: A)(implicit A: Ord[A]): Boolean =
      !Eq[Ordering].equals(A.compare(l, r), EQUAL)
  }
  case class Person(age: Int, name: String)
  object Person {
    implicit val OrdPerson: Ord[Person] = new Ord[Person] {
      def compare(l: Person, r: Person): Ordering =
        if (l.age < r.age) LT else if (l.age > r.age) GT
        else if (l.name < r.name) LT else if (l.name > r.name) GT
        else EQUAL
    }
    implicit val EqPerson: Eq[Person] = new Eq[Person] {
      def equals(l: Person, r: Person): Boolean =
        l == r
    }
  }

  //
  // EXERCISE 1
  //
  // Write a version of `sort1` called `sort2` that uses the polymorphic `List`
  // type, and which uses the `Ord` type class, including the compare syntax
  // operator `<` to compare elements.
  //
  def sort1(l: List[Int]): List[Int] = l match {
    case Nil => Nil
    case x :: xs =>
      val (lessThan, notLessThan) = xs.partition(_ < x)

      sort1(lessThan) ++ List(x) ++ sort1(notLessThan)
  }
  def sort2[A: Ord](l: List[A]): List[A] = {
    l.sortWith(implicitly[Ord[A]].compare(_, _) != GT)
  }

  //
  // EXERCISE 2
  //
  // Create a data structure and an instance of this type class for the data
  // structure.
  //
  trait PathLike[A] {
    def child(parent: A, name: String): A

    def parent(node: A): Option[A]

    def root: A
  }
  object PathLike {
    def apply[A](implicit A: PathLike[A]): PathLike[A] = A
  }

  sealed trait MyPath
  case class Node(parent: MyPath, name: String) extends MyPath
  case object RootNode extends MyPath

  object MyPath {
    implicit val MyPathPathLike: PathLike[MyPath] =
      new PathLike[MyPath] {
        def child(parent: MyPath, name: String): MyPath = {
          Node(parent, name)
        }
        def parent(node: MyPath): Option[MyPath] = node match {
          case Node(parent, _) => Some(parent)
          case RootNode => None
        }
        def root: MyPath = RootNode
      }
    }

  //
  // EXERCISE 3
  //
  // Create an instance of the `PathLike` type class for `java.io.File`.
  //
  implicit val FilePathLike: PathLike[java.io.File] = ???

//  implicit val FilePathLike: PathLike[java.io.File] = new PathLike[java.io.File] {
//    override def child(parent: File, name: String): File = new java.io.File(parent, name)
//    override def parent(node: File): Option[File] = Option(node.getParentFile)
//    override def root: File = new java.io.File("/")
//  }

  //
  // EXERCISE 4
  //
  // Create two laws for the `PathLike` type class.
  //
  trait PathLikeLaws[A] extends PathLike[A] {
    // trying to get root of parent should return none
    def law1: Boolean = parent(root) == None

    // transitivity law
    def law2(node: A, name: String, assertEquals: (A, A) => Boolean): Boolean =
      parent(child(node, name)) == node
  }

  //
  // EXERCISE 5
  //
  // Create a syntax class for path-like values with a `/` method that descends
  // into the given named node.
  //
  implicit class PathLikeSyntax[A](a: A) {
    def / (name: String)(implicit A : PathLike[A]): A =
      ???

    def parent(implicit A : PathLike[A]): Option[A] =
      ???
  }
  def root[A: PathLike]: A = PathLike[A].root

  root[MyPath] / "foo" / "bar" / "baz" // MyPath
  (root[MyPath] / "foo").parent        // Option[MyPath]

  //
  // EXERCISE 6
  //
  // Create an instance of the `Filterable` type class for `List`.
  //
  trait Filterable[F[_]] {
    def filter[A](fa: F[A], f: A => Boolean): F[A]
  }
  object Filterable {
    def apply[F[_]](implicit F: Filterable[F]): Filterable[F] = F
  }
  implicit val FilterableList: Filterable[List] = ???

  //
  // EXERCISE 7
  //
  // Create a syntax class for `Filterable` that lets you call `.filterWith` on any
  // type for which there exists a `Filterable` instance.
  //
  implicit class FilterableSyntax[F[_], A](fa: F[A]) {
    ???
  }
  // List(1, 2, 3).filterWith(_ == 2)

  //
  // EXERCISE 8
  //
  // Create an instance of the `Collection` type class for `List`.
  //
  trait Collection[F[_]] {
    def empty[A]: F[A]
    def cons[A](a: A, as: F[A]): F[A]
    def uncons[A](fa: F[A]): Option[(A, F[A])]
  }
  object Collection {
    def apply[F[_]](implicit F: Collection[F]): Collection[F] = F
  }
  implicit val ListCollection: Collection[List] = ???

  val example = Collection[List].cons(1, Collection[List].empty)

  //
  // EXERCISE 9
  //
  // Create laws for the `Collection` type class.
  //
  trait CollectionLaws[F[_]] extends Collection[F] {

  }

  //
  // EXERCISE 10
  //
  // Create syntax for values of any type that has `Collection` instances.
  // Specifically, add an `uncons` method to such types.
  //
  implicit class CollectionSyntax[F[_], A](fa: F[A]) {
    ???

    def cons(a: A)(implicit F: Collection[F]): F[A] = F.cons(a, fa)
  }
  def empty[F[_]: Collection, A]: F[A] = Collection[F].empty[A]
  // List(1, 2, 3).uncons // Some((1, List(2, 3)))
}
