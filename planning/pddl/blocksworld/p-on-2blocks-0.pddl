
(define (problem BW-rand-3)
(:domain blocksworld)
(:objects b1 b2)
(:init
(arm-empty)
(on-table b1)
(on-table b2)
(clear b1)
(clear b2)
)
(:goal
(and
(arm-empty)
(on b2 b1))
)
)


