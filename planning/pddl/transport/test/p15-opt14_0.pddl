; Automatically converted to only require STRIPS and negative preconditions

(define (problem transport-three-cities-sequential-12nodes-1000size-3degree-100mindistance-2trucks-5packages-2014seed)
  (:domain transport)
  (:objects city-1-loc-1 city-2-loc-1 city-3-loc-1 city-1-loc-2 city-2-loc-2 city-3-loc-2 city-1-loc-3 city-2-loc-3 city-3-loc-3 city-1-loc-4 city-2-loc-4 city-3-loc-4 city-1-loc-5 city-2-loc-5 city-3-loc-5 city-1-loc-6 city-2-loc-6 city-3-loc-6 city-1-loc-7 city-2-loc-7 city-3-loc-7 city-1-loc-8 city-2-loc-8 city-3-loc-8 city-1-loc-9 city-2-loc-9 city-3-loc-9 city-1-loc-10 city-2-loc-10 city-3-loc-10 city-1-loc-11 city-2-loc-11 city-3-loc-11 city-1-loc-12 city-2-loc-12 city-3-loc-12 truck-1 truck-2 package-1 package-2 package-3 package-4 package-5 capacity-0 capacity-1 capacity-2 capacity-3 capacity-4)
  (:init
    (capacity-predecessor capacity-0 capacity-1)
    (capacity-predecessor capacity-1 capacity-2)
    (capacity-predecessor capacity-2 capacity-3)
    (capacity-predecessor capacity-3 capacity-4)
    (road city-1-loc-3 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-3)
    (road city-1-loc-4 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-4)
    (road city-1-loc-5 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-5)
    (road city-1-loc-5 city-1-loc-3)
    (road city-1-loc-3 city-1-loc-5)
    (road city-1-loc-5 city-1-loc-4)
    (road city-1-loc-4 city-1-loc-5)
    (road city-1-loc-7 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-7)
    (road city-1-loc-8 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-8)
    (road city-1-loc-8 city-1-loc-4)
    (road city-1-loc-4 city-1-loc-8)
    (road city-1-loc-9 city-1-loc-1)
    (road city-1-loc-1 city-1-loc-9)
    (road city-1-loc-9 city-1-loc-4)
    (road city-1-loc-4 city-1-loc-9)
    (road city-1-loc-9 city-1-loc-7)
    (road city-1-loc-7 city-1-loc-9)
    (road city-1-loc-9 city-1-loc-8)
    (road city-1-loc-8 city-1-loc-9)
    (road city-1-loc-10 city-1-loc-7)
    (road city-1-loc-7 city-1-loc-10)
    (road city-1-loc-11 city-1-loc-2)
    (road city-1-loc-2 city-1-loc-11)
    (road city-1-loc-11 city-1-loc-3)
    (road city-1-loc-3 city-1-loc-11)
    (road city-1-loc-11 city-1-loc-6)
    (road city-1-loc-6 city-1-loc-11)
    (road city-1-loc-12 city-1-loc-3)
    (road city-1-loc-3 city-1-loc-12)
    (road city-1-loc-12 city-1-loc-6)
    (road city-1-loc-6 city-1-loc-12)
    (road city-1-loc-12 city-1-loc-7)
    (road city-1-loc-7 city-1-loc-12)
    (road city-1-loc-12 city-1-loc-11)
    (road city-1-loc-11 city-1-loc-12)
    (road city-2-loc-2 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-2)
    (road city-2-loc-4 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-4)
    (road city-2-loc-4 city-2-loc-2)
    (road city-2-loc-2 city-2-loc-4)
    (road city-2-loc-5 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-5)
    (road city-2-loc-5 city-2-loc-2)
    (road city-2-loc-2 city-2-loc-5)
    (road city-2-loc-5 city-2-loc-4)
    (road city-2-loc-4 city-2-loc-5)
    (road city-2-loc-6 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-6)
    (road city-2-loc-7 city-2-loc-2)
    (road city-2-loc-2 city-2-loc-7)
    (road city-2-loc-7 city-2-loc-3)
    (road city-2-loc-3 city-2-loc-7)
    (road city-2-loc-7 city-2-loc-4)
    (road city-2-loc-4 city-2-loc-7)
    (road city-2-loc-7 city-2-loc-5)
    (road city-2-loc-5 city-2-loc-7)
    (road city-2-loc-8 city-2-loc-3)
    (road city-2-loc-3 city-2-loc-8)
    (road city-2-loc-8 city-2-loc-7)
    (road city-2-loc-7 city-2-loc-8)
    (road city-2-loc-10 city-2-loc-1)
    (road city-2-loc-1 city-2-loc-10)
    (road city-2-loc-10 city-2-loc-2)
    (road city-2-loc-2 city-2-loc-10)
    (road city-2-loc-10 city-2-loc-4)
    (road city-2-loc-4 city-2-loc-10)
    (road city-2-loc-11 city-2-loc-2)
    (road city-2-loc-2 city-2-loc-11)
    (road city-2-loc-11 city-2-loc-5)
    (road city-2-loc-5 city-2-loc-11)
    (road city-2-loc-11 city-2-loc-9)
    (road city-2-loc-9 city-2-loc-11)
    (road city-2-loc-12 city-2-loc-9)
    (road city-2-loc-9 city-2-loc-12)
    (road city-2-loc-12 city-2-loc-11)
    (road city-2-loc-11 city-2-loc-12)
    (road city-3-loc-5 city-3-loc-3)
    (road city-3-loc-3 city-3-loc-5)
    (road city-3-loc-6 city-3-loc-4)
    (road city-3-loc-4 city-3-loc-6)
    (road city-3-loc-6 city-3-loc-5)
    (road city-3-loc-5 city-3-loc-6)
    (road city-3-loc-7 city-3-loc-4)
    (road city-3-loc-4 city-3-loc-7)
    (road city-3-loc-7 city-3-loc-6)
    (road city-3-loc-6 city-3-loc-7)
    (road city-3-loc-8 city-3-loc-5)
    (road city-3-loc-5 city-3-loc-8)
    (road city-3-loc-8 city-3-loc-6)
    (road city-3-loc-6 city-3-loc-8)
    (road city-3-loc-8 city-3-loc-7)
    (road city-3-loc-7 city-3-loc-8)
    (road city-3-loc-9 city-3-loc-2)
    (road city-3-loc-2 city-3-loc-9)
    (road city-3-loc-9 city-3-loc-4)
    (road city-3-loc-4 city-3-loc-9)
    (road city-3-loc-9 city-3-loc-6)
    (road city-3-loc-6 city-3-loc-9)
    (road city-3-loc-10 city-3-loc-1)
    (road city-3-loc-1 city-3-loc-10)
    (road city-3-loc-10 city-3-loc-3)
    (road city-3-loc-3 city-3-loc-10)
    (road city-3-loc-11 city-3-loc-1)
    (road city-3-loc-1 city-3-loc-11)
    (road city-3-loc-11 city-3-loc-10)
    (road city-3-loc-10 city-3-loc-11)
    (road city-3-loc-12 city-3-loc-2)
    (road city-3-loc-2 city-3-loc-12)
    (road city-3-loc-12 city-3-loc-4)
    (road city-3-loc-4 city-3-loc-12)
    (road city-3-loc-12 city-3-loc-9)
    (road city-3-loc-9 city-3-loc-12)
    (road city-1-loc-6 city-2-loc-10)
    (road city-2-loc-10 city-1-loc-6)
    (road city-1-loc-2 city-3-loc-10)
    (road city-3-loc-10 city-1-loc-2)
    (road city-2-loc-10 city-3-loc-10)
    (road city-3-loc-10 city-2-loc-10)
    (at package-1 city-2-loc-5)
    (at package-2 city-1-loc-2)
    (at package-3 city-2-loc-12)
    (at package-4 city-3-loc-11)
    (at package-5 city-1-loc-4)
    (at truck-1 city-2-loc-5)
    (capacity truck-1 capacity-4)
    (at truck-2 city-1-loc-11)
    (capacity truck-2 capacity-3)
    (location city-1-loc-1)
    (location city-2-loc-1)
    (location city-3-loc-1)
    (location city-1-loc-2)
    (location city-2-loc-2)
    (location city-3-loc-2)
    (location city-1-loc-3)
    (location city-2-loc-3)
    (location city-3-loc-3)
    (location city-1-loc-4)
    (location city-2-loc-4)
    (location city-3-loc-4)
    (location city-1-loc-5)
    (location city-2-loc-5)
    (location city-3-loc-5)
    (location city-1-loc-6)
    (location city-2-loc-6)
    (location city-3-loc-6)
    (location city-1-loc-7)
    (location city-2-loc-7)
    (location city-3-loc-7)
    (location city-1-loc-8)
    (location city-2-loc-8)
    (location city-3-loc-8)
    (location city-1-loc-9)
    (location city-2-loc-9)
    (location city-3-loc-9)
    (location city-1-loc-10)
    (location city-2-loc-10)
    (location city-3-loc-10)
    (location city-1-loc-11)
    (location city-2-loc-11)
    (location city-3-loc-11)
    (location city-1-loc-12)
    (location city-2-loc-12)
    (location city-3-loc-12)
    (vehicle truck-1)
    (locatable truck-1)
    (vehicle truck-2)
    (locatable truck-2)
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
    (capacity-number capacity-0)
    (capacity-number capacity-1)
    (capacity-number capacity-2)
    (capacity-number capacity-3)
    (capacity-number capacity-4)
  )
  (:goal
    (and
      (at package-1 city-1-loc-11)
    )
  )
)
