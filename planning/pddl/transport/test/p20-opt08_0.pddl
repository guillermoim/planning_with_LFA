; Automatically converted to only require STRIPS and negative preconditions

(define (problem transport-two-cities-sequential-20nodes-1000size-4degree-100mindistance-3trucks-11packages-2008seed)
  (:domain transport)
  (:objects city-1-loc-1 city-2-loc-1 city-1-loc-2 city-2-loc-2 city-1-loc-3 city-2-loc-3 city-1-loc-4 city-2-loc-4 city-1-loc-5 city-2-loc-5 city-1-loc-6 city-2-loc-6 city-1-loc-7 city-2-loc-7 city-1-loc-8 city-2-loc-8 city-1-loc-9 city-2-loc-9 city-1-loc-10 city-2-loc-10 city-1-loc-11 city-2-loc-11 city-1-loc-12 city-2-loc-12 city-1-loc-13 city-2-loc-13 city-1-loc-14 city-2-loc-14 city-1-loc-15 city-2-loc-15 city-1-loc-16 city-2-loc-16 city-1-loc-17 city-2-loc-17 city-1-loc-18 city-2-loc-18 city-1-loc-19 city-2-loc-19 city-1-loc-20 city-2-loc-20 truck-1 truck-2 truck-3 package-1 package-2 package-3 package-4 package-5 package-6 package-7 package-8 package-9 package-10 package-11 capacity-0 capacity-1 capacity-2 capacity-3 capacity-4)
  (:init
    (capacity-predecessor capacity-0 capacity-1)
    (capacity-predecessor capacity-1 capacity-2)
    (capacity-predecessor capacity-2 capacity-3)
    (capacity-predecessor capacity-3 capacity-4)
    (road city-1-loc-3 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-3)
    (road city-1-loc-4 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-4)
    (road city-1-loc-5 city-1-loc-4)
    (road city-1-loc-4 city-1-loc-5)
    (road city-1-loc-6 city-1-loc-2)
    (road city-1-loc-2 city-1-loc-6)
    (road city-1-loc-7 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-7)
    (road city-1-loc-7 city-1-loc-3)
    (road city-1-loc-3 city-1-loc-7)
    (road city-1-loc-8 city-1-loc-7)
    (road city-1-loc-7 city-1-loc-8)
    (road city-1-loc-9 city-1-loc-6)
    (road city-1-loc-6 city-1-loc-9)
    (road city-1-loc-10 city-1-loc-3)
    (road city-1-loc-3 city-1-loc-10)
    (road city-1-loc-10 city-1-loc-7)
    (road city-1-loc-7 city-1-loc-10)
    (road city-1-loc-10 city-1-loc-8)
    (road city-1-loc-8 city-1-loc-10)
    (road city-1-loc-11 city-1-loc-9)
    (road city-1-loc-9 city-1-loc-11)
    (road city-1-loc-12 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-12)
    (road city-1-loc-12 city-1-loc-3)
    (road city-1-loc-3 city-1-loc-12)
    (road city-1-loc-13 city-1-loc-9)
    (road city-1-loc-9 city-1-loc-13)
    (road city-1-loc-13 city-1-loc-11)
    (road city-1-loc-11 city-1-loc-13)
    (road city-1-loc-14 city-1-loc-4)
    (road city-1-loc-4 city-1-loc-14)
    (road city-1-loc-14 city-1-loc-5)
    (road city-1-loc-5 city-1-loc-14)
    (road city-1-loc-14 city-1-loc-8)
    (road city-1-loc-8 city-1-loc-14)
    (road city-1-loc-15 city-1-loc-9)
    (road city-1-loc-9 city-1-loc-15)
    (road city-1-loc-15 city-1-loc-11)
    (road city-1-loc-11 city-1-loc-15)
    (road city-1-loc-15 city-1-loc-13)
    (road city-1-loc-13 city-1-loc-15)
    (road city-1-loc-16 city-1-loc-11)
    (road city-1-loc-11 city-1-loc-16)
    (road city-1-loc-16 city-1-loc-13)
    (road city-1-loc-13 city-1-loc-16)
    (road city-1-loc-16 city-1-loc-15)
    (road city-1-loc-15 city-1-loc-16)
    (road city-1-loc-17 city-1-loc-8)
    (road city-1-loc-8 city-1-loc-17)
    (road city-1-loc-17 city-1-loc-10)
    (road city-1-loc-10 city-1-loc-17)
    (road city-1-loc-17 city-1-loc-11)
    (road city-1-loc-11 city-1-loc-17)
    (road city-1-loc-17 city-1-loc-15)
    (road city-1-loc-15 city-1-loc-17)
    (road city-1-loc-18 city-1-loc-9)
    (road city-1-loc-9 city-1-loc-18)
    (road city-1-loc-18 city-1-loc-13)
    (road city-1-loc-13 city-1-loc-18)
    (road city-1-loc-18 city-1-loc-15)
    (road city-1-loc-15 city-1-loc-18)
    (road city-1-loc-19 city-1-loc-12)
    (road city-1-loc-12 city-1-loc-19)
    (road city-1-loc-20 city-1-loc-2)
    (road city-1-loc-2 city-1-loc-20)
    (road city-1-loc-20 city-1-loc-18)
    (road city-1-loc-18 city-1-loc-20)
    (road city-2-loc-7 city-2-loc-3)
    (road city-2-loc-3 city-2-loc-7)
    (road city-2-loc-8 city-2-loc-6)
    (road city-2-loc-6 city-2-loc-8)
    (road city-2-loc-9 city-2-loc-2)
    (road city-2-loc-2 city-2-loc-9)
    (road city-2-loc-10 city-2-loc-4)
    (road city-2-loc-4 city-2-loc-10)
    (road city-2-loc-11 city-2-loc-3)
    (road city-2-loc-3 city-2-loc-11)
    (road city-2-loc-11 city-2-loc-9)
    (road city-2-loc-9 city-2-loc-11)
    (road city-2-loc-12 city-2-loc-4)
    (road city-2-loc-4 city-2-loc-12)
    (road city-2-loc-12 city-2-loc-10)
    (road city-2-loc-10 city-2-loc-12)
    (road city-2-loc-13 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-13)
    (road city-2-loc-13 city-2-loc-9)
    (road city-2-loc-9 city-2-loc-13)
    (road city-2-loc-14 city-2-loc-10)
    (road city-2-loc-10 city-2-loc-14)
    (road city-2-loc-14 city-2-loc-12)
    (road city-2-loc-12 city-2-loc-14)
    (road city-2-loc-15 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-15)
    (road city-2-loc-15 city-2-loc-4)
    (road city-2-loc-4 city-2-loc-15)
    (road city-2-loc-15 city-2-loc-10)
    (road city-2-loc-10 city-2-loc-15)
    (road city-2-loc-16 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-16)
    (road city-2-loc-16 city-2-loc-9)
    (road city-2-loc-9 city-2-loc-16)
    (road city-2-loc-16 city-2-loc-13)
    (road city-2-loc-13 city-2-loc-16)
    (road city-2-loc-17 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-17)
    (road city-2-loc-17 city-2-loc-10)
    (road city-2-loc-10 city-2-loc-17)
    (road city-2-loc-17 city-2-loc-14)
    (road city-2-loc-14 city-2-loc-17)
    (road city-2-loc-17 city-2-loc-15)
    (road city-2-loc-15 city-2-loc-17)
    (road city-2-loc-18 city-2-loc-3)
    (road city-2-loc-3 city-2-loc-18)
    (road city-2-loc-18 city-2-loc-7)
    (road city-2-loc-7 city-2-loc-18)
    (road city-2-loc-18 city-2-loc-11)
    (road city-2-loc-11 city-2-loc-18)
    (road city-2-loc-19 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-19)
    (road city-2-loc-19 city-2-loc-13)
    (road city-2-loc-13 city-2-loc-19)
    (road city-2-loc-19 city-2-loc-15)
    (road city-2-loc-15 city-2-loc-19)
    (road city-2-loc-19 city-2-loc-16)
    (road city-2-loc-16 city-2-loc-19)
    (road city-2-loc-19 city-2-loc-17)
    (road city-2-loc-17 city-2-loc-19)
    (road city-2-loc-20 city-2-loc-5)
    (road city-2-loc-5 city-2-loc-20)
    (road city-2-loc-20 city-2-loc-8)
    (road city-2-loc-8 city-2-loc-20)
    (road city-2-loc-20 city-2-loc-13)
    (road city-2-loc-13 city-2-loc-20)
    (road city-2-loc-20 city-2-loc-19)
    (road city-2-loc-19 city-2-loc-20)
    (road city-1-loc-12 city-2-loc-4)
    (road city-2-loc-4 city-1-loc-12)
    (at package-1 city-1-loc-10)
    (at package-2 city-1-loc-6)
    (at package-3 city-1-loc-5)
    (at package-4 city-1-loc-14)
    (at package-5 city-1-loc-13)
    (at package-6 city-1-loc-15)
    (at package-7 city-1-loc-3)
    (at package-8 city-1-loc-18)
    (at package-9 city-1-loc-8)
    (at package-10 city-1-loc-6)
    (at package-11 city-1-loc-4)
    (at truck-1 city-2-loc-11)
    (capacity truck-1 capacity-3)
    (at truck-2 city-2-loc-13)
    (capacity truck-2 capacity-3)
    (at truck-3 city-2-loc-15)
    (capacity truck-3 capacity-3)
    (location city-1-loc-1)
    (location city-2-loc-1)
    (location city-1-loc-2)
    (location city-2-loc-2)
    (location city-1-loc-3)
    (location city-2-loc-3)
    (location city-1-loc-4)
    (location city-2-loc-4)
    (location city-1-loc-5)
    (location city-2-loc-5)
    (location city-1-loc-6)
    (location city-2-loc-6)
    (location city-1-loc-7)
    (location city-2-loc-7)
    (location city-1-loc-8)
    (location city-2-loc-8)
    (location city-1-loc-9)
    (location city-2-loc-9)
    (location city-1-loc-10)
    (location city-2-loc-10)
    (location city-1-loc-11)
    (location city-2-loc-11)
    (location city-1-loc-12)
    (location city-2-loc-12)
    (location city-1-loc-13)
    (location city-2-loc-13)
    (location city-1-loc-14)
    (location city-2-loc-14)
    (location city-1-loc-15)
    (location city-2-loc-15)
    (location city-1-loc-16)
    (location city-2-loc-16)
    (location city-1-loc-17)
    (location city-2-loc-17)
    (location city-1-loc-18)
    (location city-2-loc-18)
    (location city-1-loc-19)
    (location city-2-loc-19)
    (location city-1-loc-20)
    (location city-2-loc-20)
    (vehicle truck-1)
    (locatable truck-1)
    (vehicle truck-2)
    (locatable truck-2)
    (vehicle truck-3)
    (locatable truck-3)
    (package package-1)
    (locatable package-1)
    (package package-2)
    (locatable package-2)
    (package package-3)
    (locatable package-3)
    (package package-4)
    (locatable package-4)
    (package package-5)
    (locatable package-5)
    (package package-6)
    (locatable package-6)
    (package package-7)
    (locatable package-7)
    (package package-8)
    (locatable package-8)
    (package package-9)
    (locatable package-9)
    (package package-10)
    (locatable package-10)
    (package package-11)
    (locatable package-11)
    (capacity-number capacity-0)
    (capacity-number capacity-1)
    (capacity-number capacity-2)
    (capacity-number capacity-3)
    (capacity-number capacity-4)
  )
  (:goal
    (and
      (at package-6 city-2-loc-12)
    )
  )
)